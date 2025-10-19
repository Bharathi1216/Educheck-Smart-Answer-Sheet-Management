# ...existing code...
import os
import io
import json
from pymongo import MongoClient
from google.cloud import vision
from dotenv import load_dotenv
from utils.gemini_processor import GeminiProcessor
from config.settings import settings
from pdf2image import convert_from_path

load_dotenv()

# MongoDB setup
client = MongoClient(settings.MONGO_URI)
db = client[settings.DB_NAME]

# Gemini setup - use a working model name from your available list
gemini = GeminiProcessor(settings.GEMINI_API_KEY, model_name="models/gemini-pro-latest")

def extract_text_from_file(file_path):
    """
    Returns list of pages: [{"page": int, "text": str}]
    For PDFs, convert to images and OCR per page, keeping page index.
    For images, return single page with page=1.
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
            if response and response.full_text_annotation:
                text = response.full_text_annotation.text
            pages.append({"page": idx, "text": text})
    else:
        with open(file_path, "rb") as f:
            content = f.read()
        img = vision.Image(content=content)
        response = vision_client.document_text_detection(image=img)
        text = ""
        if response and response.full_text_annotation:
            text = response.full_text_annotation.text
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
    uploads_dir = settings.UPLOAD_FOLDER or "uploads"
    for fname in os.listdir(uploads_dir):
        fpath = os.path.join(uploads_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
            continue

        print(f"Processing {fname}...")
        pages = extract_text_from_file(fpath)

        # store raw pages and page indices
        doc_type = classify_file(fname)
        doc = {
            "filename": fname,
            "filepath": fpath,
            "uploaded_at": None,
            "pages": pages
        }

        # Use Gemini to parse pages into structured parts/answers/metadata
        try:
            structured = gemini.parse_pages_to_structure(pages)
        except Exception as e:
            print("Error resolving MCQ ambiguity:", e)
            structured = {}

        # For answer keys / student answers, we may want answers mapping
        doc["structured"] = structured

        # For student answer sheets, attempt handwriting correction for a unified text blob
        try:
            all_text = "\n\n".join([p["text"] for p in pages])
            corrected = gemini.correct_handwritten_text(all_text)
        except Exception as e:
            print("Error in handwriting correction:", e)
            corrected = all_text
        doc["corrected_text"] = corrected

        # insert into appropriate collection
        db[doc_type].insert_one(doc)
        print(f"Stored {fname} as {doc_type}")

        # print extracted info for immediate feedback
        if doc_type == "answer_key":
            print("Extracted answers (answer_map):", structured.get("answer_map"))
            print("Metadata from answer key:", structured.get("metadata"))
        if doc_type == "question_paper":
            print("Question paper metadata:", structured.get("metadata"))
        if doc_type == "student_answer":
            meta = structured.get("metadata", {})
            print("Student metadata (name, roll, course):", {k: meta.get(k) for k in ("name","roll","course_code","date")})

def evaluate_student_answers():
    # naive matching: pick first answer_key document
    answer_key_doc = db.answer_key.find_one()
    if not answer_key_doc:
        print("No answer_key found; skipping evaluation.")
        return

    answer_map = answer_key_doc.get("structured", {}).get("answer_map", {})
    for student_doc in db.student_answer.find():
        student_answers = student_doc.get("structured", {}).get("answer_map", {})
        # fallback: if structured answer_map empty, try to parse from corrected_text using simple regex
        if not student_answers:
            # simple parse A/B/C letters per Qn using regex
            import re
            text = student_doc.get("corrected_text", "")
            matches = re.findall(r"(?:Q|Question)?\s*(\d+)[\)\:\-\.]?\s*([A-D])", text, re.IGNORECASE)
            student_answers = {f"Q{m[0]}": m[1].upper() for m in matches}

        try:
            eval_result = gemini.evaluate_answer(student_answers, answer_map)
        except Exception as e:
            print("Error in answer evaluation:", e)
            eval_result = {"scores": {}, "feedback": ""}

        score = gemini._calculate_weighted_score(eval_result, rubric=None)
        result_doc = {
            "student_filename": student_doc.get("filename"),
            "student_answers": student_answers,
            "answer_key_used": answer_map,
            "evaluation": eval_result,
            "score": score,
            "evaluated_at": None
        }
        db.results.insert_one(result_doc)
        print(f"Scored {student_doc.get('filename')}: {score}")

def generate_feedback_for_students():
    for result in db.results.find({"feedback_generated": {"$ne": True}}):
        eval_info = result.get("evaluation", {})
        feedback = ""
        try:
            # if we have feedback from eval, use it; else ask Gemini
            if isinstance(eval_info, dict) and eval_info.get("feedback"):
                feedback = eval_info.get("feedback")
            else:
                # ask gemini to produce a short personalized feedback
                student_answers = result.get("student_answers", {})
                answer_key = result.get("answer_key_used", {})
                prompt = (
                    "Provide a short personalized feedback for the student based on evaluation details.\n\n"
                    f"STUDENT_ANSWERS: {json.dumps(student_answers)}\nANSWER_KEY: {json.dumps(answer_key)}\n\nRespond with one paragraph."
                )
                resp = gemini.model.generate_content(prompt) if gemini.model else None
                feedback = resp.text.strip() if resp else ""
        except Exception as e:
            print("Error generating feedback:", e)
            feedback = ""

        db.results.update_one({"_id": result["_id"]}, {"$set": {"feedback": feedback, "feedback_generated": True}})
        print(f"Feedback stored for {result.get('student_filename')}: {feedback[:120]}")

# Add a small DBInstance wrapper so other modules can import db_instance.connect()
class DBInstance:
	def __init__(self, client_obj=None, db_obj=None):
		# reuse module-level client and db by default
		self._client = client_obj or client
		self._db = db_obj or db

	def connect(self):
		"""Return the pymongo Database object (for compatibility with app.py)."""
		return self._db

	def get_client(self):
		return self._client

	def get_db(self):
		return self._db

	def get_collection(self, name):
		return self._db[name]

# export instance
db_instance = DBInstance()

if __name__ == "__main__":
    process_uploads()
    evaluate_student_answers()
    generate_feedback_for_students()
# ...existing code...