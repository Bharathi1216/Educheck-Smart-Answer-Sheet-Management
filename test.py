# test_vision_key.py
import os
import base64
import requests
import sys
import argparse

# prefer google-cloud client when service-account JSON available
from google.cloud import vision
from google.oauth2 import service_account
from google.api_core import exceptions as api_exceptions

parser = argparse.ArgumentParser(description="Test Google Vision credentials with a local image")
parser.add_argument('--image', '-i', help='Path to image file', default=os.getenv('TEST_IMAGE_PATH', 'uploads/answer_key.jpg'))
parser.add_argument('--key', '-k', help='Vision API key (optional, will fallback to env GOOGLE_VISION_API_KEY)', default=os.getenv('GOOGLE_VISION_API_KEY'))
# accept explicit SA but we'll resolve from env and common locations if not provided
parser.add_argument('--sa', help='Service account JSON path (optional)', default=None)
args = parser.parse_args()

img_path = args.image
api_key = args.key
cli_sa = args.sa

if not os.path.exists(img_path):
    print(f"Error: image not found at: {img_path}")
    uploads_dir = os.path.dirname(img_path) or "uploads"
    print(f"Check the uploads folder: {os.path.abspath(uploads_dir)}")
    try:
        files = os.listdir(uploads_dir)
        if files:
            print("Files in uploads folder:")
            for f in files:
                print(" -", f)
        else:
            print("Uploads folder is empty.")
    except Exception:
        print("Could not list uploads folder.")
    print("\nPlace a test image in the uploads folder or pass --image / set TEST_IMAGE_PATH to a valid file.")
    sys.exit(1)

def resolve_service_account_path(cli_path=None):
    project_root = os.path.abspath(os.path.dirname(__file__))
    candidates = []
    if cli_path:
        candidates.append(cli_path)
    # env vars
    candidates.append(os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or "")
    candidates.append(os.getenv('GOOGLE_VISION_KEY_PATH') or "")
    # common filenames in project root
    candidates += [os.path.join(project_root, n) for n in (
        'service-account.json',
        'service-account-key.json',
        'google-vision-key.json',
        'google-credentials.json'
    )]
    # known download location (you provided one)
    candidates.append(r'c:\Users\Admin\Downloads\service-account-key.json')
    # also check absolute path provided earlier in workspace
    candidates.append(os.path.join(project_root, 'service-account.json'))
    for c in candidates:
        if not c:
            continue
        # expanduser and normalize
        c_norm = os.path.expanduser(c)
        if os.path.isabs(c_norm):
            if os.path.exists(c_norm):
                return c_norm
        else:
            # consider relative path from project root
            abs_c = os.path.abspath(os.path.join(project_root, c_norm))
            if os.path.exists(abs_c):
                return abs_c
    return None

sa_resolved = resolve_service_account_path(cli_sa)

# Prefer service-account JSON when present
if sa_resolved:
    print(f"Using service-account JSON: {sa_resolved}")
    try:
        creds = service_account.Credentials.from_service_account_file(sa_resolved)
        client = vision.ImageAnnotatorClient(credentials=creds)
        with open(img_path, "rb") as f:
            content = f.read()
        image = vision.Image(content=content)
        resp = client.document_text_detection(image=image)
        if getattr(resp, "error", None) and getattr(resp.error, "message", None):
            print("Vision client error:", resp.error.message)
            sys.exit(1)
        text = ""
        if getattr(resp, "full_text_annotation", None):
            text = resp.full_text_annotation.text or ""
        print("OCR via service-account JSON succeeded. Extracted text (truncated):")
        print(text[:1000])
        sys.exit(0)
    except Exception as e:
        print("Service-account Vision client failed:", e)
        # fall through to API key path if provided

# Fallback: use REST API key if provided
if api_key:
    print("No valid service-account found, using API key (REST) for Vision if it is valid.")
    try:
        with open(img_path, "rb") as f:
            b = base64.b64encode(f.read()).decode()
        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = {"requests":[{"image":{"content":b},"features":[{"type":"DOCUMENT_TEXT_DETECTION","maxResults":1}]}]}
        r = requests.post(url, json=payload, timeout=30)
        print(r.status_code, r.text)
        sys.exit(0)
    except Exception as e:
        print("REST API call failed:", e)
        sys.exit(1)

print("No valid credentials found. Provide either a service-account JSON (pass --sa or set GOOGLE_APPLICATION_CREDENTIALS/GOOGLE_VISION_KEY_PATH) or a valid API key (pass --key or set GOOGLE_VISION_API_KEY).")
sys.exit(1)