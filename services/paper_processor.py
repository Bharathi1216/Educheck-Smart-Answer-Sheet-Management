import os
import io
import json
import re
from datetime import datetime
from pymongo import MongoClient
from google.cloud import vision
from dotenv import load_dotenv
from pdf2image import convert_from_path

load_dotenv()

from config.settings import settings
from utils.gemini_processor import GeminiProcessor

# DB + Gemini
client = MongoClient(settings.MONGO_URI)
db = client[settings.DB_NAME]

# Use configured model IDs (env overrides)
gemini_text_model = os.getenv("GEMINI_MODEL") or getattr(settings, "GEMINI_MODEL", None) or "models/gemini-pro-latest"
gemini_image_model = os.getenv("GEMINI_IMAGE_MODEL") or getattr(settings, "GEMINI_IMAGE_MODEL", None) or "models/gemini-2.5-flash-image"

gemini = GeminiProcessor(api_key=os.getenv('GEMINI_API_KEY') or settings.GEMINI_API_KEY,
                         model_name=gemini_text_model,
                         image_model=gemini_image_model)

SUPPORTED_EXT = ('.png', '.jpg', '.jpeg', '.pdf')

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

def extract_text_from_file(file_path):
    """
    Return list of pages: [{"page": int, "text": str}]
    Uses Google Vision document_text_detection for PDFs/images.
    """
    vision_client = vision.ImageAnnotatorClient()
    ext = os.path.splitext(file_path)[1].lower()
    pages = []

    if ext == ".pdf":
        images = convert_from_path(file_path)
        for idx, image in enumerate(images, start=1):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            content = buf.getvalue()
            img = vision.Image(content=content)
            response = vision_client.document_text_detection(image=img)
            text = ""
            if response and getattr(response, "full_text_annotation", None):
                text = response.full_text_annotation.text or ""
            pages.append({"page": idx, "text": text})
    else:
        with open(file_path, "rb") as f:
            content = f.read()
        img = vision.Image(content=content)
        response = vision_client.document_text_detection(image=img)
        text = ""
        if response and getattr(response, "full_text_annotation", None):
            text = response.full_text_annotation.text or ""
        pages.append({"page": 1, "text": text})

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

def process_uploads():
    """
    Scan uploads folder, OCR -> Gemini parsing -> store docs into MongoDB.
    """
    uploads_dir = settings.UPLOAD_FOLDER or "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    for fname in os.listdir(uploads_dir):
        fpath = os.path.join(uploads_dir, fname)
        if not os.path.isfile(fpath) or not fname.lower().endswith(SUPPORTED_EXT):
            continue

        print(f"[process_uploads] Processing {fname} ...")
        try:
            pages = extract_text_from_file(fpath)  # list of {"page","text"}
        except Exception as e:
            print(f"[process_uploads] OCR error for {fname}: {e}")
            continue

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
            structured = gemini.parse_pages_to_structure(pages) or {}
        except Exception as e:
            print(f"[process_uploads] parse_pages_to_structure error: {e}")
            structured = {}

        # derive ordered_questions from structured parts (preserve hierarchy & labels)
        ordered_questions = []
        try:
            parts = structured.get("parts") or {}
            ordered_questions = _flatten_questions_from_parts(parts)
            if ordered_questions:
                # attach for question paper and answer key
                doc["ordered_questions"] = ordered_questions
        except Exception as e:
            print(f"[process_uploads] flatten parts error: {e}")
        
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

        doc = {
            "filename": fname,
            "filepath": fpath,
            "uploaded_at": datetime.utcnow(),
            "pages": pages,
            "raw_text": all_text,
            "corrected_text": corrected_text,
            "metadata": metadata,
            "structured": structured,
            "answers_original": answers_original
        }

        # For answer_key and student_answer, also align answers to ordered_questions for stable storage
        if doc_type in ("answer_key", "student_answer"):
            raw_answers = {}
            # prefer structured answer_map if present, else answers_original
            raw_answers = structured.get("answer_map", {}) or doc.get("answers_original", {}) or {}
            try:
                aligned = _align_answers_to_order(ordered_questions, raw_answers) if ordered_questions else raw_answers
                # store aligned map separately to avoid overwriting originals
                doc["answers_aligned"] = aligned
            except Exception as e:
                print(f"[process_uploads] align answers error: {e}")

        try:
            db[doc_type].insert_one(doc)
            print(f"[process_uploads] Stored {fname} into collection '{doc_type}'")
        except Exception as e:
            print(f"[process_uploads] DB insert error for {fname}: {e}")

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

def evaluate_student_answers(independent_weight=0.6):
    """
    Evaluate student_answer docs against latest answer_key.
    PRESERVE original labels exactly.
    Missing student answers stored as empty strings (no shifting).
    Handle answer-key lists for 'answer any X' and use student's first answer for scoring.
    Store full student metadata in result documents.
    """
    # pick latest question paper to get authoritative ordering; fallback to latest answer_key ordering
    qp_doc = db.question_paper.find_one(sort=[("uploaded_at", -1)])
    ak_doc = db.answer_key.find_one(sort=[("uploaded_at", -1)])
    ordered_questions = []
    if qp_doc and qp_doc.get("ordered_questions"):
        ordered_questions = qp_doc.get("ordered_questions", [])
    elif ak_doc and ak_doc.get("ordered_questions"):
        ordered_questions = ak_doc.get("ordered_questions", [])
    elif ak_doc:
        # fallback to answer_map keys if nothing else
        raw_map = ak_doc.get("structured", {}).get("answer_map", {}) or ak_doc.get("answers_original", {}) or {}
        ordered_questions = list(raw_map.keys())

    if not ordered_questions:
        print("[evaluate_student_answers] No ordered question list found; aborting evaluation.")
        return

    # Load answer_map aligned if available
    answer_map = {}
    if ak_doc:
        answer_map = ak_doc.get("answers_aligned") or (ak_doc.get("structured", {}).get("answer_map", {}) or ak_doc.get("answers_original", {}))

    # per-question marks: derive from question paper metadata or answer key metadata
    meta = (qp_doc.get("metadata") if qp_doc else {}) or (ak_doc.get("metadata") if ak_doc else {})
    total_marks_meta = None
    try:
        total_marks_meta = int(meta.get("total_marks")) if meta.get("total_marks") not in (None, "", []) else None
    except Exception:
        total_marks_meta = None

    num_questions = len(ordered_questions)
    if total_marks_meta and num_questions:
        per_q_mark = round(total_marks_meta / num_questions, 2)
        total_marks = total_marks_meta
    else:
        per_q_mark = 1.0
        total_marks = per_q_mark * num_questions

    # evaluate each student using ordered_questions
    for student_doc in db.student_answer.find():
        s_filename = student_doc.get("filename")
        # prefer aligned answers saved at upload time
        student_map = student_doc.get("answers_aligned") or student_doc.get("structured", {}).get("answer_map", {}) or student_doc.get("answers_original", {}) or {}
        # if student_map isn't aligned to ordered_questions, align now
        if set(student_map.keys()) != set(ordered_questions):
            student_map = _align_answers_to_order(ordered_questions, student_map)

        per_question_results = {}
        total_obtained = 0.0
        missed_questions = []

        # minimal student info
        student_meta = student_doc.get("metadata", {}) or {}
        student_info = {
            "name": student_meta.get("name", "") or "",
            "roll": student_meta.get("roll", "") or ""
        }

        for qid in ordered_questions:
            c_answers = answer_map.get(qid) if isinstance(answer_map, dict) else None
            # normalize key answers to list as earlier function does
            if isinstance(c_answers, list):
                key_answers = c_answers
            else:
                key_answers = _ensure_key_answers_as_list(c_answers)

            # student answer from aligned map (explicit empty string when missing)
            s_ans = student_map.get(qid, "")
            if s_ans is None:
                s_ans = ""

            detail = {
                "question_id": qid,
                "student_answer": s_ans,
                "correct_answers": key_answers,
                "awarded": 0.0,
                "max_marks": per_q_mark,
                "final_percent": 0.0,
                "feedback": "",
                "reason": ""
            }

            # missing student answer -> zero
            if s_ans == "":
                detail["reason"] = "no student answer"
                detail["feedback"] = "No answer provided."
                missed_questions.append(qid)
            elif not key_answers:
                detail["reason"] = "no answer key"
                detail["feedback"] = "No answer key provided for this question."
            else:
                # MCQ handling if keys look like MCQ letters
                is_mcq_key = all([re.fullmatch(r'[A-Da-d]', ka.strip()) for ka in key_answers if isinstance(ka, str) and ka.strip() != ""])
                if is_mcq_key:
                    matched = any(str(s_ans).strip().upper() == ka.strip().upper() for ka in key_answers)
                    if matched:
                        detail["awarded"] = float(per_q_mark)
                        detail["final_percent"] = 100.0
                        detail["feedback"] = "Correct MCQ."
                        detail["reason"] = "mcq correct"
                    else:
                        detail["awarded"] = 0.0
                        detail["final_percent"] = 0.0
                        detail["feedback"] = f"Incorrect. Expected one of: {', '.join(key_answers)}"
                        detail["reason"] = "mcq incorrect"
                else:
                    # descriptive
                    student_first = _student_first_answer(s_ans)
                    ind = gemini.evaluate_without_key(student_first)
                    ind_pct = ind.get("percent", 0.0)
                    # similarity vs all key answers -> max
                    max_sim = 0.0
                    for ka in key_answers:
                        sim = gemini.similarity_percent(student_first, ka) if ka else 0.0
                        if sim > max_sim:
                            max_sim = sim
                    final_pct = round((independent_weight * ind_pct) + ((1 - independent_weight) * max_sim), 2)
                    awarded = round((final_pct / 100.0) * per_q_mark, 2)
                    detail["final_percent"] = final_pct
                    detail["awarded"] = awarded
                    detail["feedback"] = (ind.get("feedback", "") or "") + f" Similarity(max): {max_sim}%"
                    detail["reason"] = "descriptive combined eval"

            per_question_results[qid] = detail
            total_obtained += float(detail.get("awarded", 0.0))

        total_obtained = round(total_obtained, 2)
        result_doc = {
            "student_filename": s_filename,
            "student_info": student_info,
            "per_question": per_question_results,
            "missed_questions": missed_questions,
            "total_obtained": total_obtained,
            "total_marks": total_marks,
            "percentage": round((total_obtained / total_marks) * 100, 2) if total_marks > 0 else 0,
            "evaluated_at": datetime.utcnow()
        }

        try:
            db.results.insert_one(result_doc)
            print(f"[evaluate_student_answers] Stored result for {s_filename}: {total_obtained}/{total_marks}")
        except Exception as e:
            print(f"[evaluate_student_answers] DB insert error for result {s_filename}: {e}")

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