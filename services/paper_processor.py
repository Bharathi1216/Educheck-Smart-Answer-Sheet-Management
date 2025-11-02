import os
import io
import json
import re
from datetime import datetime
from typing import TYPE_CHECKING
import zipfile
import tarfile
import shutil
import tempfile

if TYPE_CHECKING:
    from pymongo import MongoClient  # type: ignore

try:
	from pymongo import MongoClient
except Exception:
	# pymongo not installed / resolvable in this environment (silences Pylance at runtime)
	class MongoClient:
		def __init__(self, *args, **kwargs):
			raise RuntimeError(
				"pymongo is not installed or could not be imported. "
				"Install it with `pip install pymongo` or ensure your environment's dependencies are available."
			)

from google.cloud import vision
from google.oauth2 import service_account
from google.api_core import exceptions as api_exceptions
from dotenv import load_dotenv
from pdf2image import convert_from_path
import base64
import requests

load_dotenv()

from config.settings import settings
from utils.gemini_processor import GeminiProcessor

# DB + Gemini
client = MongoClient(settings.MONGO_URI)
db = client[settings.DB_NAME]

# Use configured model IDs (env overrides)
gemini_text_model = os.getenv("GEMINI_MODEL") or getattr(settings, "GEMINI_MODEL", None) or "models/gemini-pro-latest"
gemini_image_model = os.getenv("GEMINI_IMAGE_MODEL") or getattr(settings, "GEMINI_IMAGE_MODEL", None) or "models/gemini-2.5-flash-image"

# Support multiple Gemini API keys via GEMINI_API_KEYS (comma-separated) or single GEMINI_API_KEY
gemini_keys_env = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
gemini = GeminiProcessor(api_key=gemini_keys_env,
                         model_name=gemini_text_model,
                         image_model=gemini_image_model)

SUPPORTED_EXT = ('.png', '.jpg', '.jpeg', '.pdf')

# Resolve Google Vision credentials robustly (env or common filenames)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
gv_direct = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
gv_path_env = os.getenv('GOOGLE_VISION_KEY_PATH') or getattr(settings, 'GOOGLE_VISION_KEY_PATH', None)

candidates = []
if gv_direct:
    candidates.append(gv_direct)
if gv_path_env:
    candidates.append(gv_path_env)
candidates.extend([
    os.path.join(project_root, 'service-account-key.json'),
    os.path.join(project_root, 'service-account.json'),
    os.path.join(project_root, 'google-vision-key.json'),
    os.path.join(project_root, 'google-credentials.json')
])

gv_key_path = None
for c in candidates:
    if not c:
        continue
    abs_c = c if os.path.isabs(c) else os.path.abspath(os.path.join(project_root, c))
    if os.path.exists(abs_c):
        gv_key_path = abs_c
        break

# Validate service-account JSON content (must contain token_uri and client_email)
def _is_valid_service_account(path):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        # essential fields required by google.oauth2.service_account
        return bool(data.get('client_email') and data.get('token_uri'))
    except Exception:
        return False

# Module-level vision client (service-account only). Do NOT fall back to REST API key.
vision_client = None
gv_api_key = os.getenv('GOOGLE_VISION_API_KEY')  # kept for diagnostics only

if gv_key_path:
    if _is_valid_service_account(gv_key_path):
        try:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gv_key_path
            creds = service_account.Credentials.from_service_account_file(gv_key_path)
            vision_client = vision.ImageAnnotatorClient(credentials=creds)
            print(f"[paper_processor] Vision client initialized with service account: {gv_key_path}")
        except Exception as e:
            print(f"[paper_processor] Failed to initialize Vision client from {gv_key_path}: {e}")
            vision_client = None
    else:
        print(f"[paper_processor] Found service account file but it is not a valid service-account JSON: {gv_key_path}")
        print("[paper_processor] Required fields missing (client_email, token_uri). Remove invalid file or set a valid service-account JSON.")
        gv_key_path = None

# Do NOT silently use REST API key — if no service-account client, instruct user to fix it.
if not vision_client:
    print("[paper_processor] ERROR: No Google Vision service-account client available. Set GOOGLE_APPLICATION_CREDENTIALS to a valid service-account JSON with Vision API access.")
    # Note: we intentionally do NOT fall back to REST API key here.

# Helper: extract digits from identifiers like 'Q12' -> '12'
def re_digits(qid):
    m = re.search(r'(\d+)', str(qid))
    return m.group(1) if m else ""

# Helper: canonicalize labels to Qn when possible
def _canonical_qid(label):
    if not label:
        return None
    s = str(label).strip()
    m = re.search(r'(\d+)', s)
    if m:
        return f"Q{int(m.group(1))}"
    return None

# Build canonical map: canonical_id -> {"label": original_label, "answer": answer}
def _build_canonical_map(orig_map):
    canon = {}
    if not orig_map:
        return canon
    for label, ans in orig_map.items():
        # PRESERVE original labels exactly - don't force canonicalization
        if label not in canon:
            canon[label] = {"label": label, "answer": ans}
        else:
            # prefer non-empty answer
            if not canon[label].get("answer") and ans:
                canon[label] = {"label": label, "answer": ans}
    return canon

def _vision_rest_annotate_image_bytes(content_bytes, api_key):
    """Call Vision REST API images:annotate with DOCUMENT_TEXT_DETECTION and return full text (or empty)."""
    try:
        b64 = base64.b64encode(content_bytes).decode('utf-8')
        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = {
            "requests": [
                {
                    "image": {"content": b64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 1}]
                }
            ]
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("responses"):
            return ""
        r0 = data["responses"][0]
        # fullTextAnnotation may be present
        if "fullTextAnnotation" in r0 and r0["fullTextAnnotation"].get("text"):
            return r0["fullTextAnnotation"]["text"]
        # fallback to textAnnotations first entry
        ta = r0.get("textAnnotations")
        if ta and isinstance(ta, list):
            return ta[0].get("description", "") or ""
        return ""
    except Exception as e:
        raise Exception(f"Vision REST API error: {e}")

def extract_text_from_file(file_path):
    """
    Return list of pages: [{"page": int, "text": str}]
    Uses Google Vision document_text_detection via service-account client only.
    """
    ext = os.path.splitext(file_path)[1].lower()
    pages = []

    # Enforce service-account client
    if not vision_client:
        raise Exception(
            "Google Vision client not initialized. Set GOOGLE_APPLICATION_CREDENTIALS to a valid service-account JSON with Vision API access."
        )

    try:
        if ext == ".pdf":
            images = convert_from_path(file_path)
            for idx, image in enumerate(images, start=1):
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                content = buf.getvalue()
                img = vision.Image(content=content)
                response = vision_client.document_text_detection(image=img)
                if getattr(response, "error", None) and getattr(response.error, "message", None):
                    raise Exception(f"Vision API error: {response.error.message}")
                text = response.full_text_annotation.text or "" if getattr(response, "full_text_annotation", None) else ""
                pages.append({"page": idx, "text": text})
        else:
            with open(file_path, "rb") as f:
                content = f.read()
            img = vision.Image(content=content)
            response = vision_client.document_text_detection(image=img)
            if getattr(response, "error", None) and getattr(response.error, "message", None):
                raise Exception(f"Vision API error: {response.error.message}")
            text = response.full_text_annotation.text or "" if getattr(response, "full_text_annotation", None) else ""
            pages.append({"page": 1, "text": text})
    except api_exceptions.Unauthenticated as e:
        raise Exception("Vision API unauthenticated (401). Check service-account credentials/permissions.") from e
    except api_exceptions.GoogleAPICallError as e:
        raise Exception(f"Vision API call failed: {e}") from e
    except Exception:
        raise

    return pages

def classify_file(filename):
    fname = filename.lower()
    if "answer_key" in fname or "answerkey" in fname or "answer-key" in fname:
        return "answer_key"
    if "student" in fname or "answer_sheet" in fname or "submission" in fname:
        return "student_answer"
    if "question" in fname or "paper" in fname or "question_paper" in fname:
        return "question_paper"
    return "misc"

def _flatten_questions_from_parts(parts):
    """
    Flatten a structured 'parts' dict/list into an ordered list of question labels,
    preserving original labels and hierarchy. Expected input is structured['parts'] as returned by Gemini.
    Example output: ['1', '2a', '2a)i', '2a)ii', '2b', '3', ...]
    """
    ordered = []

    def walk(node, prefix=""):
        # node may be dict of questions or list of subitems
        if isinstance(node, dict):
            for k in node.keys():
                # k might be 'Q1' or '1' or '2a' etc. preserve as-is
                item = node[k]
                label = str(k).strip()
                ordered.append(label)
                # if the item contains subparts as dict/list under 'subparts' or similar, recurse
                if isinstance(item, dict):
                    # common keys that may contain nested parts
                    for subkey in ("subparts", "parts", "questions", "items"):
                        if subkey in item and item[subkey]:
                            walk(item[subkey], prefix=label)
        elif isinstance(node, list):
            for el in node:
                if isinstance(el, dict):
                    # element may have id or label
                    lid = el.get("id") or el.get("label") or el.get("question_number") or None
                    if lid:
                        ordered.append(str(lid).strip())
                    else:
                        # fallback: try to find nested questions inside element
                        for subkey in ("subparts", "parts", "questions", "items"):
                            if subkey in el and el[subkey]:
                                walk(el[subkey], prefix=prefix)
                else:
                    # primitive node -> treat as label
                    ordered.append(str(el).strip())

    # top-level parts may be dict of part_name -> questions
    if isinstance(parts, dict):
        for part_name in parts.keys():
            part = parts[part_name]
            # If part contains 'questions' dict/list, walk that; otherwise try to walk the part directly
            if isinstance(part, dict) and part.get("questions"):
                walk(part["questions"], prefix=part_name)
            else:
                walk(part, prefix=part_name)
    else:
        walk(parts)
    # dedupe while preserving order
    seen = set()
    out = []
    for q in ordered:
        if not q:
            continue
        if q not in seen:
            out.append(q)
            seen.add(q)
    return out

def _align_answers_to_order(ordered_questions, raw_answers):
    """
    Align raw_answers mapping (arbitrary labels) to ordered_questions list.
    Matching strategy:
      1) exact label match (case-insensitive)
      2) numeric prefix match: 'Q2' matches '2' or '2.' etc.
      3) fallback: leave as missing -> empty string
    Returns a dict: { ordered_label: answer_str_or_list/empty_string }
    """
    aligned = {}
    if not ordered_questions:
        return aligned

    # prepare lowercase lookup for raw_answers
    raw_map = {}
    for k, v in (raw_answers or {}).items():
        raw_map[str(k).strip().lower()] = v

    # helper to extract numeric id from label
    def numeric_key(label):
        m = re.search(r'(\d+)', str(label))
        return m.group(1) if m else None

    for oq in ordered_questions:
        oq_norm = str(oq).strip()
        oq_key = oq_norm.lower()
        ans = ""
        # 1) exact match
        if oq_key in raw_map:
            ans = raw_map[oq_key]
        else:
            # 2) try numeric match
            oq_num = numeric_key(oq_norm)
            if oq_num:
                # search raw_map keys for same numeric prefix
                for rk, rv in raw_map.items():
                    rnum = numeric_key(rk)
                    if rnum and rnum == oq_num:
                        # pick first match
                        ans = rv
                        break
        # ensure empty string rather than None
        if ans is None:
            ans = ""
        aligned[oq_norm] = ans
    return aligned

def _extract_archive_to_temp(archive_path):
    """Extract archive to a temporary directory and return its path."""
    tmpdir = tempfile.mkdtemp(prefix="pp_extract_")
    try:
        # try shutil first (auto-detect format)
        shutil.unpack_archive(archive_path, tmpdir)
        return tmpdir
    except Exception:
        # fallback to specific handlers
        try:
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, 'r') as z:
                    z.extractall(tmpdir)
                return tmpdir
            elif tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, 'r:*') as t:
                    t.extractall(tmpdir)
                return tmpdir
        except Exception as e:
            # cleanup on failure
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
            raise Exception(f"Failed to extract archive {archive_path}: {e}")
    # if nothing matched, cleanup and raise
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass
    raise Exception(f"Unsupported archive format: {archive_path}")

def _process_single_file(fpath):
    """
    Process a single file path (image/pdf) exactly as earlier process_uploads logic did.
    Returns True on success, False on OCR failure.
    """
    # Normalize to absolute path
    fpath = os.path.abspath(fpath)
    fname = os.path.basename(fpath)
    print(f"[process_uploads] Processing {fname} ...")
    print(f"[process_uploads] File absolute path: {fpath}")

    if not os.path.exists(fpath):
        print(f"[process_uploads] ERROR: File does not exist: {fpath}")
        return False

    try:
        pages = extract_text_from_file(fpath)  # list of {"page","text"}
    except Exception as e:
        print(f"[process_uploads] OCR error for {fname}: {e}")
        return False

    # Log OCR diagnostics: page count and preview of each page
    try:
        num_pages = len(pages or [])
        print(f"[process_uploads] OCR produced {num_pages} page(s) for {fname}")
        for p in pages:
            txt = (p.get("text") or "")
            preview = txt.strip().replace("\n", " ")[:200]
            print(f" - page {p.get('page')} length={len(txt)} preview={preview!r}")
    except Exception as e:
        print(f"[process_uploads] OCR logging error: {e}")

    all_text = "\n\n".join([p.get("text", "") for p in pages])
    doc_type = classify_file(fname)

    # By default empty structured & metadata
    structured = {}
    metadata = {}

    # If Gemini quota exhausted skip model work
    if getattr(gemini, "is_quota_exhausted", None) and gemini.is_quota_exhausted():
        print(f"[process_uploads] Gemini quota exhausted; skipping model parsing for {fname}")
    else:
        # Clarify responsibilities in logs: Vision did OCR; Gemini only preprocesses OCR output
        print(f"[process_uploads] Vision OCR completed for {fname}. Calling Gemini to preprocess parsed text (PARSE / NORMALIZE / CORRECT).")
        # Attempt metadata extraction and structured parse
        try:
            metadata = gemini.extract_metadata(pages) or {}
        except Exception as e:
            print(f"[process_uploads] metadata extract error: {e}")
            metadata = {}

        try:
            if doc_type == "answer_key":
                timeout_val = os.getenv("GEMINI_PER_ATTEMPT_TIMEOUT_ANSWER_KEY", "60")
                try:
                    t_sec = int(timeout_val)
                except Exception:
                    t_sec = 60
                structured = gemini.parse_pages_to_structure(pages, model_timeout=t_sec) or {}
            else:
                structured = gemini.parse_pages_to_structure(pages) or {}
        except Exception as e:
            # If model raised quota exhaustion, gemini has already set cooldown; log and fallback to OCR-only
            print(f"[process_uploads] parse_pages_to_structure error: {e}")
            structured = {}

    # If structured empty, provide informative log (helps debug "No answer provided")
    try:
        answer_map = structured.get("answer_map", {}) if isinstance(structured, dict) else {}
        answers_original = {}
        try:
            answers_original = gemini.resolve_mcq_ambiguity(all_text, original_text=all_text, prefer_regex=True) if not gemini.is_quota_exhausted() else {}
        except Exception:
            answers_original = {}
        if not answer_map and not answers_original:
            print(f"[process_uploads] WARNING: No answers found for {fname}. Likely causes:")
            print(" - OCR produced empty or low-quality text, or")
            print(" - Gemini parsing skipped/failed due to quota or model errors.")
            print(" Inspect OCR previews above and/or check Gemini quota/billing in Google Cloud Console.")
    except Exception as e:
        print(f"[process_uploads] post-parse diagnostics error: {e}")

    all_text = "\n\n".join([p.get("text", "") for p in pages])
    doc_type = classify_file(fname)

    # Metadata & structured parse (Gemini)
    metadata = {}
    try:
        metadata = gemini.extract_metadata(pages) or {}
    except Exception as e:
        print(f"[process_uploads] metadata extract error: {e}")

    structured = {}
    try:
        # If this is an answer_key or question_paper, give Gemini more time per attempt (configurable)
        if doc_type in ("answer_key", "question_paper"):
            timeout_val = os.getenv("GEMINI_PER_ATTEMPT_TIMEOUT_ANSWER_KEY", os.getenv("GEMINI_PER_ATTEMPT_TIMEOUT", "180"))
            try:
                t_sec = int(timeout_val)
            except Exception:
                t_sec = 180
            raw_parsed = gemini.parse_pages_to_structure(pages, model_timeout=t_sec) or {}
        else:
            raw_parsed = gemini.parse_pages_to_structure(pages) or {}
    except Exception as e:
        print(f"[process_uploads] parse_pages_to_structure error: {e}")
        raw_parsed = {}

    # Normalize parsed structure to guarantee consecutive keys and aligned answers
    try:
        if isinstance(raw_parsed, dict):
            structured = gemini.normalize_parsed_structure(raw_parsed)
        else:
            structured = gemini.normalize_parsed_structure({})
    except Exception as e:
        print(f"[process_uploads] structure normalization error: {e}")
        structured = raw_parsed or {}

    # Derive canonical ordered questions and answer maps from normalized structured
    ordered_questions = structured.get("ordered_questions_consecutive", [])  # ['1','2',...']
    consec_to_label = structured.get("consec_to_label", {})  # '1' -> original label
    label_to_consec = structured.get("label_to_consec", {})  # original label -> '1'
    answers_consecutive = structured.get("answers_consecutive_aligned", {}) or {}

    # Preserve original MCQ letters (regex-first)
    try:
        answers_original = gemini.resolve_mcq_ambiguity(all_text, original_text=all_text, prefer_regex=True)
    except Exception:
        answers_original = {}

    # Corrected (readability) text - do not use to overwrite MCQ letters
    try:
        corrected_text = gemini.correct_handwritten_text(all_text)
    except Exception:
        corrected_text = all_text

    # If this is a student submission, remap their extracted answers to canonical consecutive keys
    answers_aligned = {}
    answers_consecutive_aligned = {}
    missing_questions = []

    raw_answers = structured.get("answer_map_original") or structured.get("answer_map") or answers_original or {}

    try:
        # Best-effort alignment: map raw_answers -> ordered_questions
        aligned_original = _align_answers_to_order([consec_to_label.get(k, k) for k in ordered_questions], raw_answers) if ordered_questions else (raw_answers or {})
        answers_aligned = aligned_original if isinstance(aligned_original, dict) else {}

        # Build student consecutive mapping using label_to_consec
        for orig_label, ans in answers_aligned.items():
            orig_norm = str(orig_label).strip()
            consec = label_to_consec.get(orig_norm)
            if not consec:
                m = re.search(r'(\d+)', orig_norm)
                if m:
                    consec = m.group(1)
            if consec:
                answers_consecutive_aligned[str(consec)] = ans if ans is not None else ""
            else:
                # store under audit key when cannot remap
                answers_consecutive_aligned[f"orig::{orig_norm}"] = ans if ans is not None else ""

        # Ensure all canonical questions exist and mark missing
        for q in ordered_questions:
            if q not in answers_consecutive_aligned:
                answers_consecutive_aligned[q] = ""
                missing_questions.append(q)

    except Exception as e:
        print(f"[process_uploads] student answer remapping error: {e}")
        answers_aligned = raw_answers or {}
        answers_consecutive_aligned = answers_consecutive or {}
        # fill missing slots
        for q in ordered_questions:
            if q not in answers_consecutive_aligned:
                answers_consecutive_aligned[q] = ""
                if q not in missing_questions:
                    missing_questions.append(q)

    doc = {
        "filename": fname,
        "filepath": fpath,
        "uploaded_at": datetime.utcnow(),
        "pages": pages,
        "raw_text": all_text,
        "corrected_text": corrected_text,
        "metadata": metadata,
        "structured": structured,
        "answers_original": answers_original,
        "answers_aligned": answers_aligned,
        "ordered_questions_original": structured.get("ordered_questions_original", []),
        "ordered_questions_consecutive": ordered_questions,
        "label_to_consec": label_to_consec,
        "consec_to_label": consec_to_label,
        "answers_consecutive_aligned": answers_consecutive_aligned,
        "missing_questions": missing_questions  # new: list of consecutive ids missing in student answers
    }

    try:
        db[doc_type].insert_one(doc)
        print(f"[process_uploads] Stored {fname} into collection '{doc_type}' (consecutive mapping stored)")
        return True
    except Exception as e:
        print(f"[process_uploads] DB insert error for {fname}: {e}")
        return False

# Resolve uploads dir once (absolute) so all code uses same folder
UPLOADS_DIR = os.path.abspath(getattr(settings, "UPLOAD_FOLDER", "uploads") or "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

def process_uploads():
    """
    Scan uploads folder, handle archives if present -> OCR -> Gemini parsing -> store docs into MongoDB.
    """
    uploads_dir = UPLOADS_DIR
    os.makedirs(uploads_dir, exist_ok=True)

    # Show what files are present (absolute paths) to help debug path issues
    try:
        entries = os.listdir(uploads_dir)
        print(f"[process_uploads] Uploads folder: {os.path.abspath(uploads_dir)} ({len(entries)} entries)")
        for e in entries:
            print(f" - {os.path.join(os.path.abspath(uploads_dir), e)}")
    except Exception as e:
        print(f"[process_uploads] Could not list uploads folder: {e}")

    for entry in os.listdir(uploads_dir):
        entry_path = os.path.join(uploads_dir, entry)
        if os.path.isdir(entry_path):
            # skip directories at top level (or optionally walk them)
            continue

        ext = os.path.splitext(entry)[1].lower()

        # Handle archives: zip/tar/tgz/ etc.
        if ext in ('.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz'):
            try:
                extracted_dir = _extract_archive_to_temp(entry_path)
                # walk extracted dir recursively and process supported files
                for root, _, files in os.walk(extracted_dir):
                    for f in files:
                        fp = os.path.join(root, f)
                        if f.lower().endswith(SUPPORTED_EXT):
                            try:
                                _process_single_file(fp)
                            except Exception as e:
                                print(f"[process_uploads] error processing extracted file {fp}: {e}")
                # cleanup extracted files
                try:
                    shutil.rmtree(extracted_dir)
                except Exception:
                    pass
            except Exception as e:
                print(f"[process_uploads] failed to extract archive {entry_path}: {e}")
            continue

        # Normal supported files (images / pdf)
        if entry.lower().endswith(SUPPORTED_EXT):
            try:
                _process_single_file(entry_path)
            except Exception as e:
                print(f"[process_uploads] error processing {entry_path}: {e}")
            continue

        # skip other files
        print(f"[process_uploads] Skipping unsupported file: {entry}")

def _ensure_key_answers_as_list(c_ans):
    """Normalize answer key entries to list of strings for uniform handling."""
    if c_ans is None:
        return []
    if isinstance(c_ans, list):
        return [str(x).strip() for x in c_ans if x is not None]
    # sometimes key may be a comma-separated string -> split
    if isinstance(c_ans, str) and (',' in c_ans):
        parts = [p.strip() for p in c_ans.split(',') if p.strip() != ""]
        return parts
    return [str(c_ans).strip()] if str(c_ans).strip() != "" else []

def _student_first_answer(s_ans):
    """If student provided multiple answers (comma / semicolon / newline), pick first as per rule."""
    if s_ans is None:
        return ""
    if isinstance(s_ans, list) and len(s_ans) > 0:
        return str(s_ans[0]).strip()
    if isinstance(s_ans, str):
        # split on common separators but preserve original label mapping
        for sep in ['\n', ';', ',', '/']:
            if sep in s_ans:
                first = s_ans.split(sep)[0].strip()
                if first:
                    return first
        return s_ans.strip()
    return str(s_ans).strip()

def evaluate_student_answers():
    """
    Evaluates all student answers using the efficient 2-step rubric architecture.
    Step 1: Generate one rubric for the whole exam.
    Step 2: Grade each student against that rubric in a single API call.
    """
    print("[evaluate_student_answers] Starting evaluation...")
    
    # --- Load Authoritative Documents ---
    qp_doc = db.question_paper.find_one(sort=[("uploaded_at", -1)])
    ak_doc = db.answer_key.find_one(sort=[("uploaded_at", -1)])

    if not qp_doc:
        print("[evaluate_student_answers] ERROR: No question paper found in database. Aborting.")
        return
    if not ak_doc:
        print("[evaluate_student_answers] ERROR: No answer key found in database. Aborting.")
        return

    # Get the full text of the question paper and answer key
    question_paper_text = qp_doc.get("corrected_text") or qp_doc.get("raw_text", "")
    answer_key_text = ak_doc.get("corrected_text") or ak_doc.get("raw_text", "")

    # Get total marks from metadata
    meta = qp_doc.get("metadata", {}) or ak_doc.get("metadata", {})
    try:
        total_marks = int(meta.get("total_marks", 35)) # Default to 35 if not found
    except Exception:
        total_marks = 35
        
    if not question_paper_text or not answer_key_text:
        print("[evaluate_student_answers] ERROR: Question paper or answer key text is empty. Aborting.")
        return

    # --- Step 1: Generate a Single Rubric for the Entire Exam ---
    print(f"[evaluate_student_answers] Generating master rubric for {total_marks} marks...")
    rubric_data = {}
    try:
        # Check if Gemini is available and not on cooldown
        if not gemini.is_available or gemini.is_quota_exhausted():
            raise RuntimeError("Gemini is not available or quota is exhausted. Cannot generate rubric.")
            
        rubric_data = gemini.generate_rubric(
            question=question_paper_text,
            answer_key=answer_key_text,
            total_marks=total_marks
        )
        
        if not rubric_data or not rubric_data.get("rubric_items"):
            raise ValueError("Gemini returned an empty or invalid rubric.")
            
        # Convert the rubric data to a JSON string for the next prompt
        rubric_json_string = json.dumps(rubric_data, indent=2)
        print(f"[evaluate_student_answers] Master rubric generated successfully with {len(rubric_data.get('rubric_items', []))} items.")
        
    except Exception as e:
        print(f"[evaluate_student_answers] CRITICAL ERROR: Failed to generate master rubric: {e}")
        print("[evaluate_student_answers] Aborting all evaluations.")
        
   
            
def classify_file(filename):
    fname = filename.lower()
    if "answer_key" in fname or "answerkey" in fname or "answer-key" in fname:
        return "answer_key"
    if "student" in fname or "answer_sheet" in fname or "submission" in fname:
        return "student_answer"
    if "question" in fname or "paper" in fname or "question_paper" in fname:
        return "question_paper"
    return "misc"

def _flatten_questions_from_parts(parts):
    """
    Flatten a structured 'parts' dict/list into an ordered list of question labels,
    preserving original labels and hierarchy. Expected input is structured['parts'] as returned by Gemini.
    Example output: ['1', '2a', '2a)i', '2a)ii', '2b', '3', ...]
    """
    ordered = []

    def walk(node, prefix=""):
        # node may be dict of questions or list of subitems
        if isinstance(node, dict):
            for k in node.keys():
                # k might be 'Q1' or '1' or '2a' etc. preserve as-is
                item = node[k]
                label = str(k).strip()
                ordered.append(label)
                # if the item contains subparts as dict/list under 'subparts' or similar, recurse
                if isinstance(item, dict):
                    # common keys that may contain nested parts
                    for subkey in ("subparts", "parts", "questions", "items"):
                        if subkey in item and item[subkey]:
                            walk(item[subkey], prefix=label)
        elif isinstance(node, list):
            for el in node:
                if isinstance(el, dict):
                    # element may have id or label
                    lid = el.get("id") or el.get("label") or el.get("question_number") or None
                    if lid:
                        ordered.append(str(lid).strip())
                    else:
                        # fallback: try to find nested questions inside element
                        for subkey in ("subparts", "parts", "questions", "items"):
                            if subkey in el and el[subkey]:
                                walk(el[subkey], prefix=prefix)
                else:
                    # primitive node -> treat as label
                    ordered.append(str(el).strip())

    # top-level parts may be dict of part_name -> questions
    if isinstance(parts, dict):
        for part_name in parts.keys():
            part = parts[part_name]
            # If part contains 'questions' dict/list, walk that; otherwise try to walk the part directly
            if isinstance(part, dict) and part.get("questions"):
                walk(part["questions"], prefix=part_name)
            else:
                walk(part, prefix=part_name)
    else:
        walk(parts)
    # dedupe while preserving order
    seen = set()
    out = []
    for q in ordered:
        if not q:
            continue
        if q not in seen:
            out.append(q)
            seen.add(q)
    return out

def _align_answers_to_order(ordered_questions, raw_answers):
    """
    Align raw_answers mapping (arbitrary labels) to ordered_questions list.
    Matching strategy:
      1) exact label match (case-insensitive)
      2) numeric prefix match: 'Q2' matches '2' or '2.' etc.
      3) fallback: leave as missing -> empty string
    Returns a dict: { ordered_label: answer_str_or_list/empty_string }
    """
    aligned = {}
    if not ordered_questions:
        return aligned

    # prepare lowercase lookup for raw_answers
    raw_map = {}
    for k, v in (raw_answers or {}).items():
        raw_map[str(k).strip().lower()] = v

    # helper to extract numeric id from label
    def numeric_key(label):
        m = re.search(r'(\d+)', str(label))
        return m.group(1) if m else None

    for oq in ordered_questions:
        oq_norm = str(oq).strip()
        oq_key = oq_norm.lower()
        ans = ""
        # 1) exact match
        if oq_key in raw_map:
            ans = raw_map[oq_key]
        else:
            # 2) try numeric match
            oq_num = numeric_key(oq_norm)
            if oq_num:
                # search raw_map keys for same numeric prefix
                for rk, rv in raw_map.items():
                    rnum = numeric_key(rk)
                    if rnum and rnum == oq_num:
                        # pick first match
                        ans = rv
                        break
        # ensure empty string rather than None
        if ans is None:
            ans = ""
        aligned[oq_norm] = ans
    return aligned

def _extract_archive_to_temp(archive_path):
    """Extract archive to a temporary directory and return its path."""
    tmpdir = tempfile.mkdtemp(prefix="pp_extract_")
    try:
        # try shutil first (auto-detect format)
        shutil.unpack_archive(archive_path, tmpdir)
        return tmpdir
    except Exception:
        # fallback to specific handlers
        try:
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, 'r') as z:
                    z.extractall(tmpdir)
                return tmpdir
            elif tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, 'r:*') as t:
                    t.extractall(tmpdir)
                return tmpdir
        except Exception as e:
            # cleanup on failure
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
            raise Exception(f"Failed to extract archive {archive_path}: {e}")
    # if nothing matched, cleanup and raise
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass
    raise Exception(f"Unsupported archive format: {archive_path}")

def _process_single_file(fpath):
    """
    Process a single file path (image/pdf) exactly as earlier process_uploads logic did.
    Returns True on success, False on OCR failure.
    """
    # Normalize to absolute path
    fpath = os.path.abspath(fpath)
    fname = os.path.basename(fpath)
    print(f"[process_uploads] Processing {fname} ...")
    print(f"[process_uploads] File absolute path: {fpath}")

    if not os.path.exists(fpath):
        print(f"[process_uploads] ERROR: File does not exist: {fpath}")
        return False

    try:
        pages = extract_text_from_file(fpath)  # list of {"page","text"}
    except Exception as e:
        print(f"[process_uploads] OCR error for {fname}: {e}")
        return False

    # Log OCR diagnostics: page count and preview of each page
    try:
        num_pages = len(pages or [])
        print(f"[process_uploads] OCR produced {num_pages} page(s) for {fname}")
        for p in pages:
            txt = (p.get("text") or "")
            preview = txt.strip().replace("\n", " ")[:200]
            print(f" - page {p.get('page')} length={len(txt)} preview={preview!r}")
    except Exception as e:
        print(f"[process_uploads] OCR logging error: {e}")

    all_text = "\n\n".join([p.get("text", "") for p in pages])
    doc_type = classify_file(fname)

    # By default empty structured & metadata
    structured = {}
    metadata = {}

    # If Gemini quota exhausted skip model work
    if getattr(gemini, "is_quota_exhausted", None) and gemini.is_quota_exhausted():
        print(f"[process_uploads] Gemini quota exhausted; skipping model parsing for {fname}")
    else:
        # Clarify responsibilities in logs: Vision did OCR; Gemini only preprocesses OCR output
        print(f"[process_uploads] Vision OCR completed for {fname}. Calling Gemini to preprocess parsed text (PARSE / NORMALIZE / CORRECT).")
        # Attempt metadata extraction and structured parse
        try:
            metadata = gemini.extract_metadata(pages) or {}
        except Exception as e:
            print(f"[process_uploads] metadata extract error: {e}")
            metadata = {}

        try:
            if doc_type == "answer_key":
                timeout_val = os.getenv("GEMINI_PER_ATTEMPT_TIMEOUT_ANSWER_KEY", "60")
                try:
                    t_sec = int(timeout_val)
                except Exception:
                    t_sec = 60
                structured = gemini.parse_pages_to_structure(pages, model_timeout=t_sec) or {}
            else:
                structured = gemini.parse_pages_to_structure(pages) or {}
        except Exception as e:
            # If model raised quota exhaustion, gemini has already set cooldown; log and fallback to OCR-only
            print(f"[process_uploads] parse_pages_to_structure error: {e}")
            structured = {}

    # If structured empty, provide informative log (helps debug "No answer provided")
    try:
        answer_map = structured.get("answer_map", {}) if isinstance(structured, dict) else {}
        answers_original = {}
        try:
            answers_original = gemini.resolve_mcq_ambiguity(all_text, original_text=all_text, prefer_regex=True) if not gemini.is_quota_exhausted() else {}
        except Exception:
            answers_original = {}
        if not answer_map and not answers_original:
            print(f"[process_uploads] WARNING: No answers found for {fname}. Likely causes:")
            print(" - OCR produced empty or low-quality text, or")
            print(" - Gemini parsing skipped/failed due to quota or model errors.")
            print(" Inspect OCR previews above and/or check Gemini quota/billing in Google Cloud Console.")
    except Exception as e:
        print(f"[process_uploads] post-parse diagnostics error: {e}")

    all_text = "\n\n".join([p.get("text", "") for p in pages])
    doc_type = classify_file(fname)

    # Metadata & structured parse (Gemini)
    metadata = {}
    try:
        metadata = gemini.extract_metadata(pages) or {}
    except Exception as e:
        print(f"[process_uploads] metadata extract error: {e}")

    structured = {}
    try:
        # If this is an answer_key or question_paper, give Gemini more time per attempt (configurable)
        if doc_type in ("answer_key", "question_paper"):
            timeout_val = os.getenv("GEMINI_PER_ATTEMPT_TIMEOUT_ANSWER_KEY", os.getenv("GEMINI_PER_ATTEMPT_TIMEOUT", "180"))
            try:
                t_sec = int(timeout_val)
            except Exception:
                t_sec = 180
            raw_parsed = gemini.parse_pages_to_structure(pages, model_timeout=t_sec) or {}
        else:
            raw_parsed = gemini.parse_pages_to_structure(pages) or {}
    except Exception as e:
        print(f"[process_uploads] parse_pages_to_structure error: {e}")
        raw_parsed = {}

    # Normalize parsed structure to guarantee consecutive keys and aligned answers
    try:
        if isinstance(raw_parsed, dict):
            structured = gemini.normalize_parsed_structure(raw_parsed)
        else:
            structured = gemini.normalize_parsed_structure({})
    except Exception as e:
        print(f"[process_uploads] structure normalization error: {e}")
        structured = raw_parsed or {}

    # Derive canonical ordered questions and answer maps from normalized structured
    ordered_questions = structured.get("ordered_questions_consecutive", [])  # ['1','2',...']
    consec_to_label = structured.get("consec_to_label", {})  # '1' -> original label
    label_to_consec = structured.get("label_to_consec", {})  # original label -> '1'
    answers_consecutive = structured.get("answers_consecutive_aligned", {}) or {}

    # Preserve original MCQ letters (regex-first)
    try:
        answers_original = gemini.resolve_mcq_ambiguity(all_text, original_text=all_text, prefer_regex=True)
    except Exception:
        answers_original = {}

    # Corrected (readability) text - do not use to overwrite MCQ letters
    try:
        corrected_text = gemini.correct_handwritten_text(all_text)
    except Exception:
        corrected_text = all_text

    # If this is a student submission, remap their extracted answers to canonical consecutive keys
    answers_aligned = {}
    answers_consecutive_aligned = {}
    missing_questions = []

    raw_answers = structured.get("answer_map_original") or structured.get("answer_map") or answers_original or {}

    try:
        # Best-effort alignment: map raw_answers -> ordered_questions
        aligned_original = _align_answers_to_order([consec_to_label.get(k, k) for k in ordered_questions], raw_answers) if ordered_questions else (raw_answers or {})
        answers_aligned = aligned_original if isinstance(aligned_original, dict) else {}

        # Build student consecutive mapping using label_to_consec
        for orig_label, ans in answers_aligned.items():
            orig_norm = str(orig_label).strip()
            consec = label_to_consec.get(orig_norm)
            if not consec:
                m = re.search(r'(\d+)', orig_norm)
                if m:
                    consec = m.group(1)
            if consec:
                answers_consecutive_aligned[str(consec)] = ans if ans is not None else ""
            else:
                # store under audit key when cannot remap
                answers_consecutive_aligned[f"orig::{orig_norm}"] = ans if ans is not None else ""

        # Ensure all canonical questions exist and mark missing
        for q in ordered_questions:
            if q not in answers_consecutive_aligned:
                answers_consecutive_aligned[q] = ""
                missing_questions.append(q)

    except Exception as e:
        print(f"[process_uploads] student answer remapping error: {e}")
        answers_aligned = raw_answers or {}
        answers_consecutive_aligned = answers_consecutive or {}
        # fill missing slots
        for q in ordered_questions:
            if q not in answers_consecutive_aligned:
                answers_consecutive_aligned[q] = ""
                if q not in missing_questions:
                    missing_questions.append(q)

    doc = {
        "filename": fname,
        "filepath": fpath,
        "uploaded_at": datetime.utcnow(),
        "pages": pages,
        "raw_text": all_text,
        "corrected_text": corrected_text,
        "metadata": metadata,
        "structured": structured,
        "answers_original": answers_original,
        "answers_aligned": answers_aligned,
        "ordered_questions_original": structured.get("ordered_questions_original", []),
        "ordered_questions_consecutive": ordered_questions,
        "label_to_consec": label_to_consec,
        "consec_to_label": consec_to_label,
        "answers_consecutive_aligned": answers_consecutive_aligned,
        "missing_questions": missing_questions  # new: list of consecutive ids missing in student answers
    }

    try:
        db[doc_type].insert_one(doc)
        print(f"[process_uploads] Stored {fname} into collection '{doc_type}' (consecutive mapping stored)")
        return True
    except Exception as e:
        print(f"[process_uploads] DB insert error for {fname}: {e}")
        return False

# Resolve uploads dir once (absolute) so all code uses same folder
UPLOADS_DIR = os.path.abspath(getattr(settings, "UPLOAD_FOLDER", "uploads") or "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

def process_uploads():
    """
    Scan uploads folder, handle archives if present -> OCR -> Gemini parsing -> store docs into MongoDB.
    """
    uploads_dir = UPLOADS_DIR
    os.makedirs(uploads_dir, exist_ok=True)

    # Show what files are present (absolute paths) to help debug path issues
    try:
        entries = os.listdir(uploads_dir)
        print(f"[process_uploads] Uploads folder: {os.path.abspath(uploads_dir)} ({len(entries)} entries)")
        for e in entries:
            print(f" - {os.path.join(os.path.abspath(uploads_dir), e)}")
    except Exception as e:
        print(f"[process_uploads] Could not list uploads folder: {e}")

    for entry in os.listdir(uploads_dir):
        entry_path = os.path.join(uploads_dir, entry)
        if os.path.isdir(entry_path):
            # skip directories at top level (or optionally walk them)
            continue

        ext = os.path.splitext(entry)[1].lower()

        # Handle archives: zip/tar/tgz/ etc.
        if ext in ('.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz'):
            try:
                extracted_dir = _extract_archive_to_temp(entry_path)
                # walk extracted dir recursively and process supported files
                for root, _, files in os.walk(extracted_dir):
                    for f in files:
                        fp = os.path.join(root, f)
                        if f.lower().endswith(SUPPORTED_EXT):
                            try:
                                _process_single_file(fp)
                            except Exception as e:
                                print(f"[process_uploads] error processing extracted file {fp}: {e}")
                # cleanup extracted files
                try:
                    shutil.rmtree(extracted_dir)
                except Exception:
                    pass
            except Exception as e:
                print(f"[process_uploads] failed to extract archive {entry_path}: {e}")
            continue

        # Normal supported files (images / pdf)
        if entry.lower().endswith(SUPPORTED_EXT):
            try:
                _process_single_file(entry_path)
            except Exception as e:
                print(f"[process_uploads] error processing {entry_path}: {e}")
            continue

        # skip other files
        print(f"[process_uploads] Skipping unsupported file: {entry}")

def _ensure_key_answers_as_list(c_ans):
    """Normalize answer key entries to list of strings for uniform handling."""
    if c_ans is None:
        return []
    if isinstance(c_ans, list):
        return [str(x).strip() for x in c_ans if x is not None]
    # sometimes key may be a comma-separated string -> split
    if isinstance(c_ans, str) and (',' in c_ans):
        parts = [p.strip() for p in c_ans.split(',') if p.strip() != ""]
        return parts
    return [str(c_ans).strip()] if str(c_ans).strip() != "" else []

def _student_first_answer(s_ans):
    """If student provided multiple answers (comma / semicolon / newline), pick first as per rule."""
    if s_ans is None:
        return ""
    if isinstance(s_ans, list) and len(s_ans) > 0:
        return str(s_ans[0]).strip()
    if isinstance(s_ans, str):
        # split on common separators but preserve original label mapping
        for sep in ['\n', ';', ',', '/']:
            if sep in s_ans:
                first = s_ans.split(sep)[0].strip()
                if first:
                    return first
        return s_ans.strip()
    return str(s_ans).strip()

def evaluate_student_answers():
    """
    Evaluates all student answers using the efficient 2-step rubric architecture.
    Step 1: Generate one rubric for the whole exam.
    Step 2: Grade each student against that rubric in a single API call.
    """
    print("[evaluate_student_answers] Starting evaluation...")
    
    # --- Load Authoritative Documents ---
    qp_doc = db.question_paper.find_one(sort=[("uploaded_at", -1)])
    ak_doc = db.answer_key.find_one(sort=[("uploaded_at", -1)])

    if not qp_doc:
        print("[evaluate_student_answers] ERROR: No question paper found in database. Aborting.")
        return
    if not ak_doc:
        print("[evaluate_student_answers] ERROR: No answer key found in database. Aborting.")
        return

    # Get the full text of the question paper and answer key
    question_paper_text = qp_doc.get("corrected_text") or qp_doc.get("raw_text", "")
    answer_key_text = ak_doc.get("corrected_text") or ak_doc.get("raw_text", "")

    # Get total marks from metadata
    meta = qp_doc.get("metadata", {}) or ak_doc.get("metadata", {})
    try:
        total_marks = int(meta.get("total_marks", 35)) # Default to 35 if not found
    except Exception:
        total_marks = 35
        
    if not question_paper_text or not answer_key_text:
        print("[evaluate_student_answers] ERROR: Question paper or answer key text is empty. Aborting.")
        return

    # --- Step 1: Generate a Single Rubric for the Entire Exam ---
    print(f"[evaluate_student_answers] Generating master rubric for {total_marks} marks...")
    rubric_data = {}
    try:
        # Check if Gemini is available and not on cooldown
        if not gemini.is_available or gemini.is_quota_exhausted():
            raise RuntimeError("Gemini is not available or quota is exhausted. Cannot generate rubric.")
            
        rubric_data = gemini.generate_rubric(
            question=question_paper_text,
            answer_key=answer_key_text,
            total_marks=total_marks
        )
        
        if not rubric_data or not rubric_data.get("rubric_items"):
            raise ValueError("Gemini returned an empty or invalid rubric.")
            
        # Convert the rubric data to a JSON string for the next prompt
        rubric_json_string = json.dumps(rubric_data, indent=2)
        print(f"[evaluate_student_answers] Master rubric generated successfully with {len(rubric_data.get('rubric_items', []))} items.")
        
    except Exception as e:
        print(f"[evaluate_student_answers] CRITICAL ERROR: Failed to generate master rubric: {e}")
        print("[evaluate_student_answers] Aborting all evaluations.")
        return

    # --- Step 2: Grade Each Student with the Master Rubric ---
    student_docs = db.student_answer.find({"evaluation_status": {"$ne": "completed"}})
    
    for student_doc in student_docs:
        s_filename = student_doc.get("filename")
        student_answer_text = student_doc.get("corrected_text") or student_doc.get("raw_text", "")
        student_meta = student_doc.get("metadata", {}) or {}
        student_info = {
            "name": student_meta.get("name", "Unknown"),
            "roll": student_meta.get("roll", "Unknown")
        }

        if not student_answer_text.strip():
            print(f"[evaluate_student_answers] Skipping {s_filename}: No student answer text found.")
            # Store a "failed" result
            result_doc = {
                "student_filename": s_filename,
                "student_info": student_info,
                "per_question": [], # Use 'per_question' to match your old schema if needed
                "total_obtained": 0.0,
                "total_marks": total_marks,
                "percentage": 0.0,
                "evaluated_at": datetime.utcnow(),
                "feedback_summary": "Evaluation failed: No readable answer text was extracted from the file.",
                "evaluation_status": "failed_no_text"
            }
            db.results.insert_one(result_doc)
            db.student_answer.update_one({"_id": student_doc["_id"]}, {"$set": {"evaluation_status": "failed_no_text"}})
            continue

        print(f"[evaluate_student_answers] Grading {s_filename}...")
        
        try:
            # This is the ONE API call per student
            grade_json = gemini.grade_answer_with_rubric(
                question=question_paper_text,
                answer_key=answer_key_text,
                rubric_json=rubric_json_string,
                student_answer=student_answer_text
            )
            
            if not grade_json or "error" in grade_json:
                raise ValueError(f"Grading failed: {grade_json.get('error', 'Unknown error')}")

            # Prepare the final result document for MongoDB
            total_obtained = grade_json.get("total_score_awarded", 0.0)
            final_percentage = (total_obtained / total_marks) * 100 if total_marks > 0 else 0.0
            
            result_doc = {
                "student_filename": s_filename,
                "student_info": student_info,
                "per_question": grade_json.get("grading_analysis", []), # Storing the detailed breakdown
                "missed_questions": [item['concept'] for item in grade_json.get("grading_analysis", []) if item.get('score_awarded', 0) == 0],
                "total_obtained": total_obtained,
                "total_marks": total_marks,
                "percentage": round(final_percentage, 2),
                "evaluated_at": datetime.utcnow(),
                "feedback_summary": grade_json.get("general_feedback", "No feedback generated."),
                "evaluation_status": "completed"
            }
            
            db.results.insert_one(result_doc)
            db.student_answer.update_one({"_id": student_doc["_id"]}, {"$set": {"evaluation_status": "completed"}})
            print(f"[evaluate_student_answers] Stored result for {s_filename}: {total_obtained}/{total_marks} ({final_percentage:.2f}%)")

        except Exception as e:
            print(f"[evaluate_student_answers] FAILED to grade {s_filename}: {e}")
            db.student_answer.update_one({"_id": student_doc["_id"]}, {"$set": {"evaluation_status": "failed_grading"}})

    print("[evaluate_student_answers] Evaluation run finished.")
def generate_feedback_for_students():
    """
    Aggregate per-question feedback or ask Gemini for a concise personalized paragraph.
    """
    for res in db.results.find({"feedback_generated": {"$ne": True}}):
        try:
            per_q = res.get("per_question", {})
            feedback_parts = []
            for qk, info in per_q.items():
                fb = info.get("feedback")
                if fb:
                    feedback_parts.append(f"{qk}: {fb}")
            aggregated = " ".join(feedback_parts).strip()

            if not aggregated and gemini.is_available:
                prompt = (
                    "You are an assistant that provides short constructive feedback for a student's answers.\n"
                    "STUDENT_PER_QUESTION_EVAL:\n" + json.dumps(per_q, default=str) + "\n\n"
                    "Provide a one-paragraph feedback summarizing strengths and areas to improve."
                )
                resp = gemini.model.generate_content(prompt)
                aggregated = resp.text.strip() if resp and getattr(resp, "text", None) else ""

            db.results.update_one(
                {"_id": res["_id"]}, 
                {"$set": {
                    "feedback": aggregated, 
                    "feedback_generated": True, 
                    "feedback_generated_at": datetime.utcnow()
                }}
            )
            print(f"[generate_feedback_for_students] Feedback stored for {res.get('student_filename')}: {aggregated[:120]}")
        except Exception as e:
            print(f"[generate_feedback_for_students] Error generating feedback for {res.get('_id')}: {e}")

if __name__ == "__main__":
    process_uploads()
    evaluate_student_answers()
    generate_feedback_for_students()