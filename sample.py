from pathlib import Path
import base64
import glob
import os
import time

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API Keys: Reads from GEMINI_API_KEYS (comma-separated), GEMINI_API_KEY, or local list
env_keys = os.getenv("GEMINI_API_KEYS")
single_key = os.getenv("GEMINI_API_KEY")

if env_keys:
    API_KEYS = [k.strip() for k in env_keys.split(",") if k.strip()]
elif single_key:
    API_KEYS = [single_key.strip()]
else:
    API_KEYS = [
        # "YOUR_GEMINI_API_KEY_1",
    ]

# Paths and Settings
BASE_DIR = Path(__file__).resolve().parent
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")  # Recommended for best OCR accuracy
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# Prompt to extract Gujarati text
PROMPT = """
Please extract all the Gujarati text from this newspaper image.
Use OCR to read the text accurately.
Maintain the original reading order (top to bottom, left to right).
Do not add any explanations, translations, or markdown formatting.
Output only the raw Gujarati text as it appears in the image.
"""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_mime_type(image_path):
    ext = image_path.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg']: return 'image/jpeg'
    if ext == 'png': return 'image/png'
    if ext == 'webp': return 'image/webp'
    return 'image/jpeg'


def call_gemini_api(api_key, image_path):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={api_key}"
    mime_type = get_mime_type(image_path)
    base64_image = encode_image(image_path)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,  # Low temperature for factual extraction
        }
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    return response


def process_images_with_fallback():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.*")))
    current_key_idx = 0
    success_count = 0
    print(
        f"Starting extraction for {len(images)} images using {len(API_KEYS)} API keys.")
    for img_path in images:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        # Skip if already processed
        if os.path.exists(out_path):
            print(f"Skipping {filename}, already processed.")
            continue
        print(f"Processing {filename}...")
        retries = 0
        success = False
        while retries < len(API_KEYS) * 2:  # Allow looping through keys twice
            key = API_KEYS[current_key_idx]
            try:
                response = call_gemini_api(key, img_path)
                if response.status_code == 200:
                    data = response.json()
                    try:
                        text = data['candidates'][0]['content']['parts'][0][
                            'text']
                        with open(out_path, 'w', encoding='utf-8') as f:
                            f.write(text.strip())
                        print(f"  [+] Success using Key {current_key_idx + 1}")
                        success = True
                        success_count += 1
                        # Add a small delay to avoid hitting limits even on successful requests
                        time.sleep(1.5)
                        break
                    except (KeyError, IndexError):
                        print(
                            f"  [-] Unexpected response format (blocked by safety settings?): {response.text}")
                        # Move to next image, this is not a rate limit issue
                        break
                elif response.status_code == 429:  # Rate limit exhausted
                    print(
                        f"  [!] Key {current_key_idx + 1} hit rate limit (429). Rotating key...")
                    current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                    retries += 1
                    time.sleep(2)  # Brief pause before trying next key
                else:
                    print(
                        f"  [-] API Error {response.status_code}: {response.text}")
                    # Could be invalid key, bad request, etc. Try next key.
                    current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                    retries += 1
                    time.sleep(2)
            except Exception as e:
                print(f"  [-] Connection error: {e}")
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                retries += 1
                time.sleep(5)
        if not success:
            print(f"  [X] Failed to process {filename} after all retries.")
            # If all keys are rate limited, we need a longer sleep
            if retries >= len(API_KEYS):
                print(
                    "  [Zzz] All keys seem exhausted/limited. Sleeping for 30 seconds...")
                time.sleep(30)
    print(f"\nCompleted! Successfully processed {success_count} new images.")


if __name__ == "__main__":
    if not API_KEYS:
        print(
            "Error: No Gemini API keys found.\nPlease set GEMINI_API_KEY (or GEMINI_API_KEYS) in your .env file or environment."
        )
    else:
        process_images_with_fallback()
