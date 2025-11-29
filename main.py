from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
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
    # Fallback for local testing if env var is missing
    print(" Warning: GEMINI_API_KEY not found in environment.")

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

def apply_clahe(cv_img, clip_limit=1.0, tile_size=(16, 16)):
    """Applies Contrast Limited Adaptive Histogram Equalization with optimized defaults (1.0, 16x16)."""
    if cv_img is None:
        return None

    try:
        # Determine if it's a color or grayscale image
        if len(cv_img.shape) == 3:
            lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Use optimized parameters
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            cl = clahe.apply(l)
            
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            final = clahe.apply(cv_img)
            
        return final
    except Exception:
        # If any CV error occurs, return the original image
        return cv_img   

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

def normalize_page_type(raw_type):
    """
    Ensures the page type strictly matches the API requirements.
    """
    if not raw_type:
        return "Bill Detail" # Default fallback
        
    text = raw_type.lower()
    
    if "pharmacy" in text:
        return "Pharmacy"
    elif "final" in text or "summary" in text or "receipt" in text or "net payable" in text:
        return "Final Bill"
    else:
        # Default value
        return "Bill Detail"

async def process_single_page(pil_image, page_number):
    # 1. Vision Processing 
    try:
        cv_img = pil_to_cv2(pil_image)

        enhanced_cv = apply_clahe(cv_img)
        striped_cv = apply_signal_zebra(enhanced_cv)
        final_pil = optimize_image(cv2_to_pil(striped_cv))

    except Exception as e:
        print(f"CV Error on page {page_number}: {e}")
        final_pil = optimize_image(pil_image)

    # 2. Gemini Processing
    prompt = """
    The image has been digitally enhanced for maximum contrast and row separation.
    Analyze this medical bill page and extract data in JSON format.
    
    TASK 1: CLASSIFY PAGE TYPE
    Classify this page into EXACTLY one of these three categories:
    - "Pharmacy": If the page lists medicines, tablets, syrups, batch numbers, or expiry dates.
    - "Final Bill": If the page shows the "Net Payable Amount", "Amount in Words", "Advance Paid", or is a summary receipt.
    - "Bill Detail": If the page lists hospital services like "Room Rent", "Lab Test", "Consultation", "Nursing Charges".
    
    TASK 2: EXTRACT ITEMS
    For each line item in the main table, extract:
    - "item_name": (string) Exact text description.
    - "item_quantity": (float) Quantity.
    - "item_rate": (float) Unit price/rate.
    - "item_amount": (float) Net amount for that line.
    - make sure not to extract "subtotal", "tota" or things like "amount due" as line items
    
    OUTPUT JSON STRUCTURE:
    {
      "page_type": "String (one of the 3 allowed types)",
      "items": [ ... ],
    }
    """
    
    try:
        response = await asyncio.to_thread(model.generate_content, [prompt, final_pil])
        
        # Verify response validity
        if not response.text:
            raise ValueError("Gemini returned empty response (Safety Filter?).")
        
        raw_text = response.text.strip()
        
        # 1. Remove Markdown code blocks
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        raw_text = raw_text.strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback: Sometimes Gemini uses single quotes instead of double quotes
            import ast
            try:
                data = ast.literal_eval(raw_text)
            except:
                raise ValueError(f"Failed to parse JSON. Raw output: {raw_text[:100]}...")
            
        # data = json.loads(response.text)
        
        raw_type = data.get("page_type", "Bill Detail")
        strict_page_type = normalize_page_type(raw_type)

        # Format Output
        clean_items = []
        for item in data.get("items", []) or []:
            try:
                item_amount_clean = float(item.get("item_amount", 0.0))
            except (ValueError, TypeError):
                item_amount_clean = 0.0 # Safety net for bad string inputs
        
            try:
                item_rate_clean = float(item.get("item_rate", 0.0))
            except (ValueError, TypeError):
                item_rate_clean = 0.0

            clean_items.append({
                "item_name": item.get("item_name"),
                "item_amount": item_amount_clean,
                "item_rate": item_rate_clean,
                "item_quantity": float(item.get("item_quantity", 0.0))                
            })

        usage = response.usage_metadata
        return {
            "tokens": {
                "input": usage.prompt_token_count,
                "output": usage.candidates_token_count,
                "total": usage.total_token_count
            },
            "page_no": str(page_number),
            "page_type": strict_page_type,
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
            "error": error_msg  
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

        response = await asyncio.to_thread(requests.get, request.document, headers=headers, timeout=15)

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

        final_pages = []
        total_items_count = 0

        total_tokens = sum(p["tokens"]["total"] for p in pagewise_results)
        input_tokens = sum(p["tokens"]["input"] for p in pagewise_results)
        output_tokens = sum(p["tokens"]["output"] for p in pagewise_results)
        
        for p in pagewise_results:
            total_items_count += len(p["bill_items"])
            
            # --- STRICT SCHEMA ENFORCEMENT ---
            # Only include the 3 specific fields required by the problem statement.
            final_pages.append({
                "page_no": p["page_no"],
                "page_type": p["page_type"],
                "bill_items": p["bill_items"]
            })

        return {
            "is_success": True,
            "token_usage": {
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            },
            "data": {
                "pagewise_line_items": final_pages,
                "total_item_count": total_items_count
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