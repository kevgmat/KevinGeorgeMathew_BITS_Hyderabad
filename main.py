from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import google.generativeai as genai
import PIL.Image
import io
import json
import time
import os
import asyncio
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import rotate
from pdf2image import convert_from_bytes

# --- 1. CONFIGURATION ---
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

genai.configure(api_key="AIzaSyCX4Ckj-Y9x6zuAqK8suB_Ug8mYtOGTQII", transport="rest")

model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config={"response_mime_type": "application/json"}
)

app = FastAPI(title="BFHL Datathon Invoice Extractor")

# --- 2. COMPUTER VISION LOGIC (The "Zebra" Differentiator) ---

def determine_score(arr, angle):
    """Calculates row distinctness score for deskewing."""
    data = rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1)
    return np.var(histogram)

def deskew_image_logic(image):
    """
    Optimized Deskew: Rotates image to maximize horizontal row distinctness.
    """
    h, w = image.shape[:2]
    resize_factor = 1000.0 / w if w > 1000 else 1.0
    small = cv2.resize(image, (int(w * resize_factor), int(h * resize_factor)))
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Coarse Search (-10 to 10)
    angles = np.arange(-10, 11, 1)
    scores = [determine_score(thresh, a) for a in angles]
    best_angle = angles[np.argmax(scores)]

    # Fine Search
    fine_angles = np.arange(best_angle - 1, best_angle + 1, 0.1)
    fine_scores = [determine_score(thresh, a) for a in fine_angles]
    final_angle = fine_angles[np.argmax(fine_scores)]

    # Rotate
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def apply_signal_zebra(cv_img):
    """
    Applies Signal Processing to find rows and draws zebra stripes.
    Input: OpenCV Image (BGR) -> Output: OpenCV Image (BGR)
    """
    # 1. Deskew
    img = deskew_image_logic(cv_img)
    H, W = img.shape[:2]
    
    # 2. Adaptive Threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    # 3. Remove Vertical Lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    detected_cols = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    clean_thresh = cv2.subtract(thresh, detected_cols)

    # 4. Signal Processing (Projection Profile)
    row_sum = np.sum(clean_thresh, axis=1)
    window_size = 15
    smoothed_signal = np.convolve(row_sum, np.ones(window_size)/window_size, mode='same')
    
    # 5. Find Separators
    inverted_signal = -smoothed_signal
    peaks, _ = find_peaks(inverted_signal, distance=20, prominence=50)
    separators = [0] + list(peaks) + [H]
    separators.sort()
    
    # 6. Draw Stripes
    overlay = img.copy()
    bands_drawn = 0
    
    for i in range(len(separators) - 1):
        top = separators[i]
        bottom = separators[i+1]
        band_height = bottom - top
        
        if band_height > 10:
            if bands_drawn % 2 == 0:
                color = (220, 255, 255) 
            else:
                color = (255, 240, 220) 

            cv2.rectangle(overlay, (0, top), (W, bottom), color, -1)
            bands_drawn += 1

    # 7. Blend
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

def cv2_to_pil(cv_img):
    """Helper to convert OpenCV back to PIL for Gemini."""
    return PIL.Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_img):
    """Helper to convert PIL to OpenCV."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- 3. PROCESSING LOGIC ---

def optimize_image(pil_image):
    """Resize image to max 1024px width to reduce latency."""
    if pil_image.width > 1024:
        aspect_ratio = pil_image.height / pil_image.width
        new_size = (1024, int(1024 * aspect_ratio))
        return pil_image.resize(new_size, PIL.Image.Resampling.LANCZOS)
    return pil_image

async def process_single_page(pil_image, page_number):
    """
    Process one page:
    1. Apply Zebra Stripes (CV2)
    2. Resize (PIL)
    3. Send to Gemini (API)
    """
    # A. Apply Differentiator (Zebra Stripes)
    try:
        cv_img = pil_to_cv2(pil_image)
        striped_cv_img = apply_signal_zebra(cv_img)
        processed_pil = cv2_to_pil(striped_cv_img)
    except Exception as e:
        print(f"Warning: Zebra logic failed on page {page_number}, utilizing raw image. Error: {e}")
        processed_pil = pil_image

    # B. Optimize for Latency
    final_img = optimize_image(processed_pil)
    
    # C. Anchor Prompt (Leverages the Stripes)
    prompt = """
    Analyze this medical bill page row by row. 
    The image has alternating colored bands to help you separate rows.

    Return a JSON OBJECT with a key "items" containing a list.
    For each row/band in the table:
    1. "raw_text_anchor": Quote the full text line exactly as seen in the band.
    2. "item_name": Extract the Description.
    3. "item_quantity": Extract the Qty (number).
    4. "item_rate": Extract the Rate (number).
    5. "item_amount": Extract the Amount (number).

    Rules:
    - Exclude headers, patient details, and sub-totals.
    - If a column is missing, use null.
    """

    try:
        response = await asyncio.to_thread(model.generate_content, [prompt, final_img])
        data = json.loads(response.text)
        
        # Cleanup Structure
        items = data.get("items", [])
        if isinstance(data, list): items = data
        
        # Clean Output (Remove anchor text to save bandwidth/match schema)
        clean_items = []
        for item in items:
            clean_items.append({
                "item_name": item.get("item_name"),
                "item_quantity": item.get("item_quantity", 1.0),
                "item_rate": item.get("item_rate"),
                "item_amount": item.get("item_amount")
            })

        return {
            "page_no": str(page_number),
            "bill_items": clean_items
        }
    except Exception as e:
        print(f"Error on page {page_number}: {e}")
        return {
            "page_no": str(page_number),
            "bill_items": []
        }

# --- 4. API ENDPOINT ---

@app.post("/extract-bill")
async def extract_bill(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        file_content = await file.read()
        filename = file.filename.lower()
        pagewise_results = []

        # 1. Load Images (PDF or Single Image)
        if filename.endswith(".pdf"):
            images = convert_from_bytes(file_content)
            # Create async tasks for all pages
            tasks = [process_single_page(img, i + 1) for i, img in enumerate(images)]
            pagewise_results = await asyncio.gather(*tasks)
        else:
            img = PIL.Image.open(io.BytesIO(file_content))
            pagewise_results = [await process_single_page(img, 1)]

        # 2. Aggregation & Math
        total_items_count = 0
        global_reconciled_amount = 0.0

        for page_data in pagewise_results:
            items = page_data.get("bill_items", [])
            total_items_count += len(items)
            
            # Python Math Validation
            page_total = sum(float(item.get("item_amount") or 0) for item in items)
            global_reconciled_amount += page_total

        # 3. Final Response
        return JSONResponse(content={
            "is_success": True,
            "data": {
                "pagewise_line_items": pagewise_results,
                "total_item_count": total_items_count,
                "reconciled_amount": round(global_reconciled_amount, 2)
            },
            "latency": round(time.time() - start_time, 2)
        })

    except Exception as e:
        return JSONResponse(content={
            "is_success": False,
            "error": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)