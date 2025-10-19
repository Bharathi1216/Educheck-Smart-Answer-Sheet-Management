# This file is a snapshot backup created before integrating enhanced matching/evaluation logic.
# Keep it as a restore point. Do not import or execute this file in production.

import os
import io
import json
import re
from datetime import datetime
from pymongo import MongoClient
from google.cloud import vision
from dotenv import load_dotenv
from pdf2image import convert_from_path

from utils.gemini_processor import GeminiProcessor

load_dotenv()

from config.settings import settings

# DB + Gemini
client = MongoClient(settings.MONGO_URI)
db = client[settings.DB_NAME]
gemini = GeminiProcessor(os.getenv('GEMINI_API_KEY') or settings.GEMINI_API_KEY, model_name="gemini-1.0-pro")

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
            "uploaded_at": None,
            "pages": pages,
            "raw_text": all_text,
            "corrected_text": corrected_text,
            "metadata": metadata,
            "structured": structured,
            "answers_original": answers_original
        }

        try:
            db[doc_type].insert_one(doc)
            print(f"[process_uploads] Stored {fname} into collection '{doc_type}'")
        except Exception as e:
            print(f"[process_uploads] DB insert error for {fname}: {e}")

def evaluate_student_answers(independent_weight=0.6):
    """
    Evaluate student_answer docs against latest answer_key.
    PRESERVE original question labels exactly - no canonicalization that changes numbers.
    Missing student answers are treated as empty (no shifting).
    """
    answer_key_doc = db.answer_key.find_one(sort=[("uploaded_at", -1)])
    if not answer_key_doc:
        print("[evaluate_student_answers] No answer_key found; aborting.")
        return

    ak_struct = answer_key_doc.get("structured", {}) or {}
    raw_answer_map = ak_struct.get("answer_map", {}) or answer_key_doc.get("answers_original", {}) or {}

    # PRESERVE original labels - don't canonicalize
    answer_map = {}
    for label, answer in raw_answer_map.items():
        if answer is None:
            answer_map[label] = ""  # Ensure empty string for missing answers
        else:
            answer_map[label] = answer

    # determine question order and per-question marks
    ak_meta = ak_struct.get("metadata", {}) or answer_key_doc.get("metadata", {}) or {}
    total_marks_meta = None
    try:
        total_marks_meta = int(ak_meta.get("total_marks")) if ak_meta.get("total_marks") not in (None, "", []) else None
    except Exception:
        total_marks_meta = None

    # Extract ordered questions from parts structure if available
    ordered_questions = []
    parts = ak_struct.get("parts", {})
    for part_name in sorted(parts.keys()):
        part_data = parts[part_name]
        if isinstance(part_data, dict):
            ordered_questions.extend(sorted(part_data.keys()))
        elif isinstance(part_data, list):
            for item in part_data:
                if isinstance(item, dict) and item.get("id"):
                    ordered_questions.append(item["id"])

    # Fallback: use answer_map keys sorted naturally
    if not ordered_questions:
        ordered_questions = sorted(answer_map.keys(), key=lambda x: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(x))])

    num_questions = len(ordered_questions)
    if total_marks_meta and num_questions:
        per_q_mark = round(total_marks_meta / num_questions, 2)
        total_marks = total_marks_meta
    else:
        per_q_mark = 1.0
        total_marks = per_q_mark * num_questions

    # iterate students
    for student_doc in db.student_answer.find():
        s_filename = student_doc.get("filename")
        s_struct = student_doc.get("structured", {}) or {}
        raw_student_map = s_struct.get("answer_map", {}) or student_doc.get("answers_original", {}) or {}

        # PRESERVE original student answers exactly
        student_map = {}
        for label, answer in raw_student_map.items():
            if answer is None:
                student_map[label] = ""  # Ensure empty string for missing answers
            else:
                student_map[label] = answer

        per_question_results = {}
        total_obtained = 0.0
        missed_questions = []

        for qid in ordered_questions:
            c_ans = answer_map.get(qid, "")
            s_ans = student_map.get(qid, "")

            detail = {
                "question_id": qid,
                "student_answer": s_ans,
                "correct_answer": c_ans,
                "awarded": 0.0,
                "max_marks": per_q_mark,
                "final_percent": 0.0,
                "feedback": "",
                "reason": ""
            }

            # CRITICAL: Handle missing answers - award zero, no shifting
            if s_ans == "":
                detail["reason"] = "no student answer"
                detail["feedback"] = "No answer provided."
                detail["awarded"] = 0.0
                missed_questions.append(qid)
            elif c_ans == "":
                detail["reason"] = "no answer key"
                detail["feedback"] = "No answer key provided for this question."
                detail["awarded"] = 0.0
            else:
                # MCQ exact-match
                if re.fullmatch(r'[A-Da-d]', str(c_ans).strip()):
                    if re.fullmatch(r'[A-Da-d]', str(s_ans).strip()):
                        if str(s_ans).strip().upper() == str(c_ans).strip().upper():
                            detail["awarded"] = float(per_q_mark)
                            detail["final_percent"] = 100.0
                            detail["feedback"] = "Correct MCQ."
                            detail["reason"] = "mcq correct"
                        else:
                            detail["awarded"] = 0.0
                            detail["final_percent"] = 0.0
                            detail["feedback"] = f"Incorrect. Expected {c_ans}."
                            detail["reason"] = "mcq incorrect"
                    else:
                        # student answered textually; independent eval
                        ind = gemini.evaluate_without_key(s_ans)
                        pct = ind.get("percent", 0.0)
                        detail["independent_eval"] = ind
                        detail["final_percent"] = pct
                        detail["awarded"] = round((pct / 100.0) * per_q_mark, 2)
                        detail["feedback"] = ind.get("feedback", "")
                        detail["reason"] = "mcq key present but student text answer - independent eval"
                else:
                    # descriptive: two-step evaluation
                    ind = gemini.evaluate_without_key(s_ans)
                    ind_pct = ind.get("percent", 0.0)
                    sim_pct = gemini.similarity_percent(s_ans, c_ans)
                    
                    detail["independent_eval"] = ind
                    detail["similarity_percent"] = sim_pct
                    
                    # Use the function parameter for weighting
                    final_pct = round((independent_weight * ind_pct) + ((1 - independent_weight) * sim_pct), 2)
                    awarded = round((final_pct / 100.0) * per_q_mark, 2)
                    
                    detail["final_percent"] = final_pct
                    detail["awarded"] = awarded
                    detail["feedback"] = f"{ind.get('feedback', '')} Similarity: {sim_pct}%"
                    detail["reason"] = "descriptive combined eval"

            per_question_results[qid] = detail
            total_obtained += float(detail["awarded"])

        total_obtained = round(total_obtained, 2)
        
        # Enhanced result with complete student info
        result_doc = {
            "student_filename": s_filename,
            "student_doc_id": student_doc.get("_id"),
            "student_info": student_doc.get("metadata", {}),
            "per_question": per_question_results,
            "missed_questions": missed_questions,
            "total_obtained": total_obtained,
            "total_marks": total_marks,
            "percentage": round((total_obtained / total_marks) * 100, 2) if total_marks > 0 else 0,
            "final_feedback": " ".join([v.get("feedback", "") for v in per_question_results.values()])[:400],
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