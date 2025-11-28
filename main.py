from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import PIL.Image
import io
import json
import time
import os
import asyncio
import requests
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import rotate
from pdf2image import convert_from_bytes

# --- 1. CONFIGURATION ---
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("No GEMINI_API_KEY found in environment variables")

genai.configure(api_key=api_key, transport="rest")

model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config={"response_mime_type": "application/json"}
)

app = FastAPI(title="BFHL Datathon Invoice Extractor")

# Input Schema
class BillRequest(BaseModel):
    document: str

# --- 2. COMPUTER VISION LOGIC ---
def determine_score(arr, angle):

    data = rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1)
    return np.var(histogram)

def deskew_image_logic(image):

    h, w = image.shape[:2]
    resize_factor = 1000.0 / w if w > 1000 else 1.0
    small = cv2.resize(image, (int(w * resize_factor), int(h * resize_factor)))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    angles = np.arange(-5, 6, 1)
    scores = [determine_score(thresh, a) for a in angles]
    best_angle = angles[np.argmax(scores)]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)

    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def apply_signal_zebra(cv_img):

    img = deskew_image_logic(cv_img)
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    detected_cols = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    clean_thresh = cv2.subtract(thresh, detected_cols)

    row_sum = np.sum(clean_thresh, axis=1)
    smoothed_signal = np.convolve(row_sum, np.ones(15)/15, mode='same')
    peaks, _ = find_peaks(-smoothed_signal, distance=20, prominence=50)
    
    if len(peaks) < 3: return img # Guardrail

    separators = [0] + list(peaks) + [H]
    separators.sort()

    overlay = img.copy()
    bands_drawn = 0

    for i in range(len(separators) - 1):
        top = separators[i]
        bottom = separators[i+1]
        if (bottom - top) > 10:
            color = (220, 255, 255) if bands_drawn % 2 == 0 else (255, 240, 220)
            cv2.rectangle(overlay, (0, top), (W, bottom), color, -1)
            bands_drawn += 1
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    return img

def pil_to_cv2(pil_img):

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img):

    return PIL.Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# --- 3. PROCESSING ---
def optimize_image(pil_image):

    if pil_image.width > 1024:
        aspect_ratio = pil_image.height / pil_image.width
        new_size = (1024, int(1024 * aspect_ratio))
        return pil_image.resize(new_size, PIL.Image.Resampling.LANCZOS)
    return pil_image

async def process_single_page(pil_image, page_number):
    # 1. Vision Processing (Try/Except to prevent CV crashes)
    try:
        cv_img = pil_to_cv2(pil_image)
        striped_cv = apply_signal_zebra(cv_img)
        final_pil = optimize_image(cv2_to_pil(striped_cv))
    except Exception as e:
        print(f"CV Error on page {page_number}: {e}")
        final_pil = optimize_image(pil_image)

    # 2. Gemini Processing
    prompt = """
    Analyze this medical bill page. Return JSON with key "items".
    For each row, extract: item_name, item_quantity, item_rate, item_amount.
    Also identify the "page_type" (e.g. Pharmacy, Bill).
    """
    
    try:
        # This is where the error likely happens
        response = await asyncio.to_thread(model.generate_content, [prompt, final_pil])
        
        # Verify response validity
        if not response.text:
            raise ValueError("Gemini returned empty response (Safety Filter?).")
            
        data = json.loads(response.text)
        
        # Format Output
        clean_items = []
        for item in data.get("items", []) or []:
            clean_items.append({
                "item_name": item.get("item_name"),
                "item_quantity": item.get("item_quantity", 1.0),
                "item_rate": item.get("item_rate"),
                "item_amount": item.get("item_amount")
            })
            
        return {
            "page_no": str(page_number),
            "page_type": data.get("page_type", "Unknown"),
            "bill_items": clean_items,
            "error": None
        }

    except Exception as e:
        # --- CRITICAL: Return the actual error message ---
        error_msg = str(e)
        print(f"API Error on page {page_number}: {error_msg}")
        return {
            "page_no": str(page_number),
            "page_type": "Error",
            "bill_items": [],
            "error": error_msg  # <--- This will show in your JSON
        }

# --- 4. ENDPOINT ---
@app.post("/extract-bill-data")
async def extract_bill_data(request: BillRequest):
    start_time = time.time()
    try:
        # 1. Download
        print(f"Downloading: {request.document}")
        # Add User-Agent to avoid 403 Forbidden from some servers
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(request.document, headers=headers, timeout=15)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Download failed: Status {response.status_code}")
            
        file_content = response.content
        url_lower = request.document.lower().split('?')[0] # Ignore query params
        
        # 2. Process
        pagewise_results = []
        if url_lower.endswith(".pdf") or file_content[:4] == b'%PDF':
            images = convert_from_bytes(file_content)
            tasks = [process_single_page(img, i + 1) for i, img in enumerate(images)]
            pagewise_results = await asyncio.gather(*tasks)
        else:
            try:
                img = PIL.Image.open(io.BytesIO(file_content))
                pagewise_results = [await process_single_page(img, 1)]
            except IOError:
                raise HTTPException(status_code=400, detail="File is not a valid Image or PDF")

        # 3. Check for specific errors in pages
        errors = [p["error"] for p in pagewise_results if p.get("error")]
        if errors:
            # If errors exist, return them in the message so you can debug
            return JSONResponse(content={
                "is_success": False,
                "message": f"Processing Errors: {'; '.join(errors)}"
            }, status_code=500)

        # 4. Aggregation (The Missing Logic)
        total_items = sum(len(p["bill_items"]) for p in pagewise_results)
        
        # Calculate Total Sum (Reconciled Amount)
        total_amount = 0.0
        for p in pagewise_results:
            for item in p["bill_items"]:
                amount = item.get("item_amount")
                # Safety check: ensure amount is a number
                if amount and isinstance(amount, (int, float)):
                    total_amount += float(amount)
                elif amount and isinstance(amount, str):
                    # Cleanup string like "$1,200.50" -> 1200.50
                    try:
                        clean_amt = amount.replace('$', '').replace(',', '').strip()
                        total_amount += float(clean_amt)
                    except:
                        pass
        
        # Clean up internal keys
        final_pages = [{k: v for k, v in p.items() if k != 'error'} for p in pagewise_results]

        return {
            "is_success": True,
            "data": {
                "pagewise_line_items": final_pages,
                "total_item_count": total_items,
                "reconciled_amount": round(total_amount, 2)
            }
        }

    except Exception as e:
        return JSONResponse(content={
            "is_success": False,
            "message": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)