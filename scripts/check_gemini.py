"""
Quick Gemini Pro diagnostic.

Usage:
  (venv) PS D:\EduCheckBackend1> python scripts\check_gemini.py
  Optionally set env vars beforehand:
    $env:GEMINI_API_KEY = "..."         (Windows PowerShell)
    export GEMINI_API_KEY="..."         (bash)
    set GEMINI_MODEL="models/gemini-pro-latest"  # optional

The script will attempt a tiny model.generate_content call and print detailed error info
and guidance for common errors: 401, 404, 429, PERMISSION_DENIED, INVALID_ARGUMENT, timeouts.
"""
import os
import json
import sys
import traceback
from dotenv import load_dotenv  # added

try:
    import google.generativeai as genai
except Exception as e:
    print("ERROR: google-generativeai package not installed in the current environment.")
    print("Run: pip install google-generativeai")
    sys.exit(2)

# ensure .env from project root is loaded so GEMINI_API_KEY in .env is available
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

def classify_and_explain(exc_text, exc_type_name):
    text = (exc_text or "").lower()
    reasons = []
    if "401" in text or "unauthenticated" in text or "invalid authentication" in text:
        reasons.append(("401 UNAUTHENTICATED", "API key invalid/missing. Ensure GEMINI_API_KEY set to a valid key from Google AI Studio."))
    if "404" in text or "not_found" in text or "model does not exist" in text:
        reasons.append(("404 NOT_FOUND", "Model name is incorrect or not available. Check GEMINI_MODEL and use a valid model id (e.g. 'models/gemini-1.5-pro-latest' or 'models/gemini-pro-latest')."))
    if "429" in text or "resource_exhausted" in text or "quota" in text:
        reasons.append(("429 RESOURCE_EXHAUSTED", "Rate limit or quota exceeded. Check Google Cloud Console -> Quotas or wait and retry."))
    if "permission" in text or "permission_denied" in text:
        reasons.append(("PERMISSION_DENIED", "API key/project lacks permission for the requested model. Enable required APIs / ensure key belongs to project with access."))
    if "invalid argument" in text or "invalid_argument" in text or "400" in text:
        reasons.append(("400 INVALID_ARGUMENT", "Request payload/format invalid. Verify the prompt and SDK usage match docs."))
    if "504" in text or "timed out" in text or "timeout" in text:
        reasons.append(("504/Timeout", "Request timed out. Network, model responsiveness, or large prompt might be the cause. Retry with smaller prompt or increase timeouts."))
    if not reasons:
        reasons.append((exc_type_name, "Unknown error. See the raw exception below."))
    return reasons

def run_diagnostic():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API") or os.getenv("GEMINI_KEY")
    model_name = os.getenv("GEMINI_MODEL") or "models/gemini-pro-latest"
    if not api_key:
        print(f"GEMINI_API_KEY not set in environment or .env (checked {env_path}). Please set it and retry.")
        print("Example (PowerShell): $env:GEMINI_API_KEY = 'YOUR_KEY'")
        return 2

    print(f"Using model: {model_name}")
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print("Failed to configure google.generativeai with provided key:")
        print(str(e))
        return 3

    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print("Failed to create GenerativeModel instance. Error:")
        print(type(e).__name__, str(e))
        # still try a minimal generation to capture server-side errors if any
        # but constructing model failed (likely wrong model name or SDK issue)
        return 4

    prompt = "Say 'ok' in a single word."
    print("Sending a minimal test prompt to Gemini... (short prompt)")
    try:
        resp = model.generate_content(prompt)
        # Attempt to print concise output
        text = getattr(resp, "text", None)
        print("\nSUCCESS: Received response from Gemini.")
        if text is not None:
            print("Model output (truncated):")
            print(text.strip()[:400])
        else:
            # Some SDK responses may embed content differently
            try:
                print("Response object:", json.dumps(resp.__dict__, default=str)[:800])
            except Exception:
                print("Response repr:", repr(resp)[:800])
        return 0
    except Exception as e:
        exc_text = str(e)
        exc_type_name = type(e).__name__
        print("\nFAILED: Gemini call raised an exception.")
        print(f"Exception type: {exc_type_name}")
        print("Exception message (truncated):")
        print(exc_text[:1000])
        print("\nClassified causes & suggested fixes:")
        causes = classify_and_explain(exc_text, exc_type_name)
        for code, advice in causes:
            print(f"- {code}: {advice}")
        print("\nFull traceback (for debugging):")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_diagnostic())
