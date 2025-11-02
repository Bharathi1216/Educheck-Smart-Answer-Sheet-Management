"""
Simple validator for Google service-account JSON files.

Usage:
  python scripts\check_service_account.py
  python scripts\check_service_account.py --set     # sets env var in this process (useful before running other scripts from same shell)
  python scripts\check_service_account.py --test-image uploads/answer_key.jpg  # attempt a small OCR test (requires google-cloud-vision & network)
"""
import os
import json
import argparse
import sys

try:
    from google.oauth2 import service_account
except Exception:
    service_account = None

try:
    from google.cloud import vision
    HAS_VISION = True
except Exception:
    vision = None
    HAS_VISION = False

def candidate_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cands = []
    # Explicit env vars
    cands.append(os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or "")
    cands.append(os.getenv('GOOGLE_VISION_KEY_PATH') or "")
    # Common filenames in project
    cands += [
        os.path.join(project_root, 'service-account-key.json'),
        os.path.join(project_root, 'service-account.json'),
        os.path.join(project_root, 'google-vision-key.json'),
        os.path.join(project_root, 'google-credentials.json'),
    ]
    # Common user download path on Windows (your environment)
    cands.append(r'c:\Users\Admin\Downloads\service-account-key.json')
    # Normalize and yield unique
    seen = set()
    for p in cands:
        if not p:
            continue
        pnorm = os.path.expanduser(p)
        if not os.path.isabs(pnorm):
            pnorm = os.path.abspath(os.path.join(project_root, pnorm))
        if pnorm in seen:
            continue
        seen.add(pnorm)
        yield pnorm

def validate_sa(path):
    try:
        if not os.path.exists(path):
            return False, "not found"
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        missing = []
        for key in ('client_email', 'token_uri', 'private_key'):
            if not data.get(key):
                missing.append(key)
        if missing:
            return False, f"missing fields: {', '.join(missing)}"
        # Try to construct credentials (this does not contact network)
        if service_account is None:
            return True, "JSON looks valid (google-auth not installed to validate fully)"
        try:
            creds = service_account.Credentials.from_service_account_file(path)
        except Exception as e:
            return False, f"service_account.Credentials failed: {e}"
        # Try to create Vision client object if available (no network call here)
        if HAS_VISION:
            try:
                _ = vision.ImageAnnotatorClient(credentials=creds)
            except Exception as e:
                return False, f"vision client init failed: {e}"
        return True, "valid service-account JSON"
    except Exception as e:
        return False, f"error reading/parsing JSON: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', action='store_true', help='Set GOOGLE_APPLICATION_CREDENTIALS to the first valid path in this process')
    parser.add_argument('--test-image', help='Optional image path to run a small OCR request (requires network & Vision enabled)')
    args = parser.parse_args()

    found_any = False
    print("Checking common service-account JSON locations...")
    for p in candidate_paths():
        ok, msg = validate_sa(p)
        if ok:
            print(f"FOUND VALID: {p} -> {msg}")
            found_any = True
            if args.set:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = p
                print(f"Set GOOGLE_APPLICATION_CREDENTIALS={p} for this process.")
            break
        else:
            if os.path.exists(p):
                print(f"INVALID  : {p} -> {msg}")
            else:
                print(f"Missing  : {p}")

    if not found_any:
        print("\nNo valid service-account JSON found in common locations.")
        print("Place your JSON outside the repo, then set environment variable:")
        print("  PowerShell: $env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\\path\\to\\key.json'")
        print("  bash: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
        sys.exit(2)

    # Optional test OCR
    if args.test_image:
        if not HAS_VISION:
            print("\nCannot run OCR test: google-cloud-vision is not installed in this environment.")
            sys.exit(0)
        img = args.test_image
        if not os.path.exists(img):
            print(f"\nTest image not found: {img}")
            sys.exit(1)
        try:
            creds = service_account.Credentials.from_service_account_file(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
            client = vision.ImageAnnotatorClient(credentials=creds)
            with open(img, "rb") as fh:
                content = fh.read()
            image = vision.Image(content=content)
            resp = client.document_text_detection(image=image)
            if getattr(resp, "error", None) and getattr(resp.error, "message", None):
                print("OCR request returned error:", resp.error.message)
                sys.exit(1)
            text = ""
            if getattr(resp, "full_text_annotation", None):
                text = resp.full_text_annotation.text or ""
            elif resp.text_annotations:
                text = resp.text_annotations[0].description or ""
            print("\nOCR test succeeded. Extracted text (first 400 chars):")
            print(text[:400].replace('\n', ' '))
            sys.exit(0)
        except Exception as e:
            print("\nOCR test failed:", e)
            sys.exit(1)

    print("\nValidation completed. If a valid key was found you can now re-run your paper processor.")
    sys.exit(0)

if __name__ == '__main__':
    main()
