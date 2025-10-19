import google.generativeai as genai
import os
import json
from datetime import datetime
import re
import pymongo
import difflib
from dotenv import load_dotenv

load_dotenv()

class GeminiProcessor:
    """
    Handles Gemini AI operations:
      - parse pages into structured parts/questions/answers & metadata
      - handwriting correction / spelling normalization
      - evaluation / scoring (returns numeric scores + feedback)
    """
     
    def __init__(self, api_key=None, model_name=None, image_model=None):
        # Use explicit env or fallbacks to recommended model IDs
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        model_name = model_name or os.getenv("GEMINI_MODEL") or "models/gemini-pro-latest"
        image_model = image_model or os.getenv("GEMINI_IMAGE_MODEL") or "models/gemini-2.5-flash-image"

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # create text model if supported
                try:
                    self.model = genai.GenerativeModel(model_name)
                    self.model_name = model_name
                    self.image_model = image_model
                    self.is_available = True
                    print(f"✅ Gemini AI initialized successfully (text model: {model_name})")
                except Exception as e:
                    print(f"Gemini text model init error for '{model_name}':", e)
                    # try fallback to a safer default
                    try:
                        fallback = "models/gemini-pro-latest"
                        self.model = genai.GenerativeModel(fallback)
                        self.model_name = fallback
                        self.image_model = image_model
                        self.is_available = True
                        print(f"✅ Gemini AI initialized with fallback model {fallback}")
                    except Exception as e2:
                        print("Gemini model fallback init error:", e2)
                        self.model = None
                        self.is_available = False
            except Exception as e:
                print("Gemini configure error:", e)
                self.model = None
                self.is_available = False
        else:
            print("No GEMINI_API_KEY configured; Gemini model disabled.")
            self.model = None
            self.is_available = False

        # default rubric
        self.rubrics = {
            "default": {
                "meaning_comprehension": 40,
                "key_concepts": 30,
                "technical_accuracy": 20,
                "structure": 10
            }
        }

    def resolve_mcq_ambiguity(self, text_blob, original_text=None, prefer_regex=True, question_context=None):
        """
        Unified MCQ extractor (backwards-compatible).
        """
        if not text_blob and not original_text:
            return {}

        src = original_text or text_blob or ""
        # Regex-first extraction (safe): patterns like "Q1. A", "1) A", "Question 1: A", "1. A"
        try:
            matches = re.findall(r"(?:Q(?:uestion)?\.?\s*)?0*([0-9]{1,3})[\)\.\:\-]?\s*([A-D])\b", src, re.IGNORECASE)
            out = {f"Q{m[0]}": m[1].upper() for m in matches}
            if out or prefer_regex:
                return out
        except Exception as e:
            print("MCQ regex extraction error:", e)

        # Model-based extraction only if regex produced nothing and model exists
        if self.model:
            prompt = (
                "Return only JSON mapping question ids to single-letter MCQ answers, e.g. {\"Q1\":\"A\",\"Q2\":\"C\"}.\n\n"
                f"TEXT:\n{src}\n\nRespond only with JSON."
            )
            try:
                resp = self.model.generate_content(prompt)
                text = resp.text.strip()
                start = text.find("{")
                end = text.rfind("}") + 1
                json_blob = text[start:end] if start != -1 and end != -1 else text
                parsed = json.loads(json_blob)
                out = {}
                for k, v in parsed.items():
                    key = str(k).strip()
                    val = str(v).strip().upper() if v is not None else ""
                    out[key] = val
                return out
            except Exception as e:
                print("Error extracting MCQ via Gemini:", e)

        return {}

    def extract_metadata(self, pages_list):
        """
        Enhanced metadata extraction for database storage
        """
        combined = "\n\n".join([f"--- PAGE {p['page']} ---\n{p['text']}" for p in pages_list])
        
        meta = {
            "course_code": None,
            "total_marks": None, 
            "subject": None,
            "date": None,
            "name": None,
            "roll": None,
            "exam_type": None,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Enhanced regex patterns
        try:
            # Course code patterns
            cc_patterns = [
                r"\b([A-Z]{2,6}\s?-?\s?\d{2,4}[A-Z]?)\b",
                r"Course\s*[Cc]ode\s*[:\-]?\s*([A-Z0-9\-]+)",
                r"Subject\s*[Cc]ode\s*[:\-]?\s*([A-Z0-9\-]+)"
            ]
            
            for pattern in cc_patterns:
                cc = re.search(pattern, combined)
                if cc:
                    meta["course_code"] = cc.group(1).strip()
                    break

            tm = re.search(r"(?:Total\s*Marks|Out\s*of|Max(?:imum)?\s*Marks)\s*[:\-]?\s*(\d{1,4})", combined, re.IGNORECASE)
            if tm:
                meta["total_marks"] = tm.group(1).strip()

            dt = re.search(r"(?:Date|Dated)\s*[:\-]?\s*([0-9]{1,2}[\/\-\s][0-9]{1,2}[\/\-\s][0-9]{2,4})", combined, re.IGNORECASE)
            if dt:
                meta["date"] = dt.group(1).strip()

            name = re.search(r"\bName\s*[:\-]?\s*([A-Za-z \.\-]{3,100})", combined, re.IGNORECASE)
            if name:
                meta["name"] = name.group(1).strip()

            roll = re.search(r"\b(?:Roll|Roll\s*No|Reg(?:istration)?\s*No)\s*[:\-]?\s*([A-Za-z0-9\-\/]{3,30})", combined, re.IGNORECASE)
            if roll:
                meta["roll"] = roll.group(1).strip()

            subj = re.search(r"\bSubject\s*[:\-]?\s*([A-Za-z0-9 \-\&]{3,100})", combined, re.IGNORECASE)
            if subj:
                meta["subject"] = subj.group(1).strip()
                
        except Exception as e:
            print("Metadata regex error:", e)

        # Use model for missing critical metadata
        if self.model and (not meta["course_code"] or not meta["roll"]):
            prompt = (
                "Extract exam metadata. Return JSON with: course_code, total_marks, subject, date, name, roll, exam_type.\n"
                "Use null for missing values. Be accurate with roll numbers and course codes.\n\n"
                f"{combined}\n\nRespond only with JSON."
            )
            try:
                resp = self.model.generate_content(prompt)
                text = resp.text.strip()
                start = text.find("{")
                end = text.rfind("}") + 1
                json_blob = text[start:end] if start != -1 and end != -1 else text
                parsed = json.loads(json_blob)
                
                # Update only if we found better values
                for key in meta.keys():
                    if key in parsed and parsed[key] not in (None, "", []):
                        meta[key] = parsed[key]
                        
            except Exception as e:
                print("Gemini metadata parse error:", e)

        return meta

    def parse_pages_to_structure(self, pages_list):
        """
        Preserve original question numbers and handle missing answers
        """
        concatenated = "\n\n".join([f"--- PAGE {p['page']} ---\n{p['text']}" for p in pages_list])
        
        prompt = (
            "You are a strict JSON-only assistant. Parse the OCR output of an exam paper into a JSON structure. "
            "CRITICAL RULES:\n"
            "1) PRESERVE ORIGINAL QUESTION NUMBERS EXACTLY AS WRITTEN - DO NOT RENUMBER\n"
            "2) For missing answers, use empty string \"\" \n"
            "3) IGNORE completely any struck-out/crossed-out/overwritten answers\n"
            "4) For 'write any two' type questions in answer key: include ALL correct answers\n"
            "5) For 'write any two' in student answers: take only the FIRST answer written\n\n"
            "Tasks:\n"
            "1) Extract metadata: course_code, total_marks, subject, date, name, roll (use null if absent)\n"
            "2) Detect Parts/Subparts (e.g., Part A, Part B, A(i), (a)) and map questions under them\n"
            "3) For each question record: id (EXACT ORIGINAL like '1', '2a', 'Q3'), page number, raw_text (short), and answer\n"
            "4) For answer keys: include all possible answers for 'any X' type questions\n"
            "5) For student answers: take only first answer for 'any X' type questions\n\n"
            "Return only valid JSON with keys: metadata, parts, answer_map.\n\nOCR_TEXT:\n"
            f"{concatenated}\n\nRespond only with JSON."
        )

        if not self.model:
            return self._simple_parse_fallback(pages_list)

        try:
            resp = self.model.generate_content(prompt)
            text = resp.text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            json_blob = text[start:end] if start != -1 and end != -1 else text
            parsed = json.loads(json_blob)
            
            # Ensure missing answers are empty strings, not null
            answer_map = parsed.get("answer_map", {})
            for qid, answer in answer_map.items():
                if answer is None:
                    answer_map[qid] = ""
            
            return parsed
        except Exception as e:
            print("Error parsing pages to structure:", e)
            return self._simple_parse_fallback(pages_list)

    def correct_handwritten_text(self, extracted_text, writing_quality="medium"):
        """
        Clean OCR text but PRESERVE original question numbering exactly
        """
        if not self.model:
            return extracted_text
            
        prompt = (
            "Clean and normalize the OCR text with STRICT RULES:\n"
            "1. FIX spelling & punctuation ONLY\n" 
            "2. PRESERVE ORIGINAL QUESTION NUMBERS EXACTLY - DO NOT CHANGE FORMAT\n"
            "3. Remove ONLY struck-out or overwritten answers (ignore them completely)\n"
            "4. Keep answers region-tagged when present\n"
            "5. For missing answers, leave them as empty\n\n"
            "OCR_TEXT:\n"
            f"{extracted_text}\n\n"
            "Return only the cleaned text (no explanations)."
        )
        try:
            resp = self.model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            print("Error in handwriting correction:", e)
            return extracted_text

    def evaluate_answer(self, student_answer, correct_answer, rubric=None, subject="general"):
        """
        Evaluate and return dict:
          { "scores": {"meaning_comprehension": int,...}, "feedback": "..." , "raw": "<model output>" }
        """
        if rubric is None:
            rubric = self.rubrics.get("default", {})
        if not self.model:
            return {"scores": {}, "feedback": "", "raw": ""}

        prompt = (
            "Evaluate the student answer against the answer key. Provide numeric 0-100 scores for these dimensions: "
            "meaning_comprehension, key_concepts, technical_accuracy, structure. Also provide a concise feedback string. "
            "Return only valid JSON with keys: scores (dict), feedback (string).\n\n"
            f"STUDENT_ANSWER:\n{student_answer}\n\nANSWER_KEY:\n{correct_answer}\n\nRespond only with JSON."
        )
        try:
            resp = self.model.generate_content(prompt)
            text = resp.text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            json_blob = text[start:end] if start != -1 and end != -1 else text
            parsed = json.loads(json_blob)
            # ensure numeric scores present
            scores = parsed.get("scores", {})
            return {"scores": scores, "feedback": parsed.get("feedback", ""), "raw": text}
        except Exception as e:
            print("Error in answer evaluation:", e)
            return {"scores": {}, "feedback": "", "raw": ""}

    def _validate_correction(self, original_text, corrected_text):
        """Basic validator for corrections - accept by default."""
        return True

    def _calculate_weighted_score(self, evaluation, rubric):
        """Calculate weighted overall score based on rubric; handle rubric None."""
        if rubric is None:
            rubric = self.rubrics.get("default", {})
        if not rubric:
            rubric = {"meaning_comprehension": 1}
        total_weight = sum(rubric.values())
        weighted_score = 0
        for dimension, weight in rubric.items():
            score = 0
            if isinstance(evaluation, dict):
                score = evaluation.get("scores", {}).get(dimension, 0)
            weighted_score += (score or 0) * weight
        return round(weighted_score / total_weight, 2)

    def generate_handwriting_feedback(self, original_text, corrected_text, writing_quality="medium"):
        """Quick placeholder feedback (can be expanded)."""
        if not self.model:
            return "Handwriting feedback not available."
        prompt = (
            "Provide a one-line feedback on handwriting quality and legibility. Original and corrected text are provided.\n\n"
            f"ORIGINAL:\n{original_text}\n\nCORRECTED:\n{corrected_text}\n\nRespond with one short sentence."
        )
        try:
            resp = self.model.generate_content(prompt)
            return resp.text.strip()
        except Exception:
            return "Handwriting feedback not available."

    def evaluate_and_store_final(self, structure, student_answers, answer_key, rubric=None, mongo_uri=None, db_name="educheck", collection="results", model_weight=0.7):
        """
        Two-step evaluation with PROPER missing answer handling
        """
        if rubric is None:
            rubric = self.rubrics.get("default", {})

        # Flatten questions from parts deterministically
        parts = structure.get("parts", {}) if isinstance(structure, dict) else {}
        q_list = []
        for part_name in sorted(parts.keys()):
            questions = parts.get(part_name)
            if isinstance(questions, dict):
                for qid in sorted(questions.keys(), key=lambda x: int(re.sub(r'\D','',x) or 0)):
                    q_list.append(qid)
            elif isinstance(questions, list):
                for q in questions:
                    if isinstance(q, dict) and "id" in q:
                        q_list.append(q["id"])

        # ensure unique ordered questions
        seen = set()
        ordered_qs = []
        for q in q_list:
            if q not in seen:
                ordered_qs.append(q)
                seen.add(q)

        num_questions = len(ordered_qs)

        # determine per-question marks
        total_marks_meta = None
        try:
            tm = structure.get("metadata", {}).get("total_marks")
            total_marks_meta = int(tm) if tm not in (None, "", []) else None
        except Exception:
            total_marks_meta = None

        if total_marks_meta and num_questions:
            per_q_mark = round(total_marks_meta / num_questions, 2)
            total_marks = total_marks_meta
        else:
            per_q_mark = 1.0
            total_marks = round(per_q_mark * max(1, num_questions), 2)

        per_question_results = {}
        total_obtained = 0.0

        for qid in ordered_qs:
            # CRITICAL: Explicit empty string for missing answers - NO SHIFTING
            s_ans = student_answers.get(qid, "")
            if s_ans is None:
                s_ans = ""
            c_ans = answer_key.get(qid, "")
            if c_ans is None:
                c_ans = ""

            detail = {
                "student_answer": s_ans,
                "correct_answer": c_ans,
                "model_eval": None,
                "similarity_percent": 0.0,
                "final_percent": 0.0,
                "awarded": 0.0,
                "max_marks": per_q_mark,
                "feedback": "",
                "reason": ""
            }

            # Handle missing answers - AWARD ZERO, NO SHIFTING
            if s_ans == "":
                detail["reason"] = "no student answer"
                detail["feedback"] = "No answer submitted."
                obtained = 0.0
            elif c_ans == "":
                detail["reason"] = "no answer key" 
                detail["feedback"] = "Answer key unavailable for this question."
                obtained = 0.0
            else:
                # If MCQ (single-letter) handle sharply
                if isinstance(c_ans, str) and re.fullmatch(r"[A-Da-d]", c_ans.strip()):
                    if isinstance(s_ans, str) and re.fullmatch(r"[A-Da-d]", s_ans.strip()):
                        if s_ans.strip().upper() == c_ans.strip().upper():
                            obtained = float(per_q_mark)
                            detail["reason"] = "mcq correct"
                            detail["final_percent"] = 100.0
                            detail["feedback"] = "Correct MCQ."
                        else:
                            obtained = 0.0
                            detail["reason"] = "mcq incorrect"
                            detail["final_percent"] = 0.0
                            detail["feedback"] = "Incorrect MCQ."
                        detail["model_eval"] = {"method": "mcq_direct"}
                        detail["similarity_percent"] = 100.0 if obtained == per_q_mark else 0.0
                    else:
                        obtained = 0.0
                        detail["reason"] = "mcq format mismatch or incorrect"
                        detail["feedback"] = "Answer not in MCQ format or incorrect."
                else:
                    # Descriptive: 1) model evaluation (liberal but strict), 2) similarity
                    model_percent = 0.0
                    model_eval_raw = None
                    try:
                        # Only call model eval when model exists; evaluate_answer handles no-model case
                        model_eval_raw = self.evaluate_answer(s_ans, c_ans, rubric=rubric)
                        model_percent = self._calculate_weighted_score(model_eval_raw, rubric)
                        detail["model_eval"] = model_eval_raw
                    except Exception as e:
                        model_percent = 0.0
                        detail["model_eval"] = {"error": str(e)}

                    # similarity percent via difflib
                    try:
                        # use lowercase stripped strings
                        s_norm = str(s_ans).strip().lower()
                        c_norm = str(c_ans).strip().lower()
                        sim_ratio = difflib.SequenceMatcher(None, s_norm, c_norm).ratio()
                        sim_percent = round(sim_ratio * 100, 2)
                        detail["similarity_percent"] = sim_percent
                    except Exception:
                        detail["similarity_percent"] = 0.0

                    # combine (model more weight for liberal teacher-like grading)
                    sim_weight = 1.0 - float(model_weight)
                    final_percent = round((float(model_weight) * float(model_percent)) + (sim_weight * float(detail["similarity_percent"])), 2)
                    detail["final_percent"] = final_percent

                    # convert percent to marks
                    obtained = round((final_percent / 100.0) * per_q_mark, 2)

                    # feedback combining model feedback and similarity hint
                    model_fb = ""
                    if isinstance(model_eval_raw, dict):
                        model_fb = model_eval_raw.get("feedback", "") or ""
                    # craft short dynamic feedback
                    if obtained == 0:
                        detail["feedback"] = (model_fb + " ").strip() + f" Low similarity ({detail['similarity_percent']}%)."
                        detail["reason"] = "descriptive evaluated - low score"
                    elif obtained < per_q_mark:
                        detail["feedback"] = (model_fb + " ").strip() + f" Partial credit based on content similarity ({detail['similarity_percent']}%)."
                        detail["reason"] = "descriptive evaluated - partial"
                    else:
                        detail["feedback"] = (model_fb + " ").strip() + " Full credit."
                        detail["reason"] = "descriptive evaluated - full"

            # finalize numbers
            try:
                obtained = float(obtained)
            except Exception:
                obtained = 0.0
            obtained = round(obtained, 2)
            detail["awarded"] = obtained
            per_question_results[qid] = detail
            total_obtained += obtained

        total_obtained = round(total_obtained, 2)

        # Enhanced result payload with complete metadata
        result_payload = {
            "metadata": structure.get("metadata", {}),
            "per_question": per_question_results,
            "total_obtained": total_obtained,
            "total_marks": total_marks,
            "evaluated_at": datetime.utcnow().isoformat() + "Z",
            "student_info": {
                "roll_number": structure.get("metadata", {}).get("roll"),
                "name": structure.get("metadata", {}).get("name"), 
                "course_code": structure.get("metadata", {}).get("course_code"),
                "subject": structure.get("metadata", {}).get("subject"),
                "exam_date": structure.get("metadata", {}).get("date")
            }
        }

        # Save to MongoDB Atlas if mongo_uri provided
        if mongo_uri:
            try:
                client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                db = client[db_name]
                coll = db[collection]
                coll.insert_one(result_payload)
                client.close()
                result_payload["saved_to_db"] = True
            except Exception as e:
                result_payload["saved_to_db"] = False
                result_payload["db_error"] = str(e)

        return result_payload

    def evaluate_without_key(self, student_answer, rubric=None):
        """
        Ask the model to evaluate the quality of the student answer on a 0-100 scale,
        and return {"percent": float, "feedback": str}.
        Fallback to a simple heuristic when model not available.
        """
        if rubric is None:
            rubric = self.rubrics.get("default", {})
        # If no model, basic heuristic
        if not self.model:
            # heuristic: length and presence of keywords -> rough percent
            txt = (student_answer or "").strip()
            if not txt:
                return {"percent": 0.0, "feedback": "No answer provided."}
            ln = len(txt)
            percent = min(80.0, max(10.0, (min(ln, 800) / 800.0) * 70.0 + 10.0))
            return {"percent": round(percent, 2), "feedback": "Heuristic evaluation (no model)."}

        prompt = (
            "Return only JSON: {\"percent\":<0-100 number>, \"feedback\":\"short feedback\"}.\n\n"
            "Task: Evaluate the quality of the following student answer as a standalone response. "
            "Consider clarity, completeness, relevance and technical correctness. Provide an overall percent (0-100) "
            "and a one-line feedback.\n\n"
            f"STUDENT_ANSWER:\n{student_answer}\n\nRespond only with JSON."
        )
        try:
            resp = self.model.generate_content(prompt)
            text = resp.text.strip()
            # extract first JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            json_blob = text[start:end] if start != -1 and end != -1 else text
            parsed = json.loads(json_blob)
            percent = float(parsed.get("percent", 0) or 0)
            feedback = parsed.get("feedback", "") or ""
            return {"percent": round(percent, 2), "feedback": feedback}
        except Exception as e:
            # fallback heuristic if model output invalid
            print("evaluate_without_key error:", e)
            return {"percent": 0.0, "feedback": "Evaluation failed (model error)."}

    def similarity_percent(self, student_answer, correct_answer):
        """
        Returns a percentage 0-100 representing semantic similarity.
        Uses Gemini model when available; otherwise falls back to difflib ratio.
        """
        a = (student_answer or "").strip()
        b = (correct_answer or "").strip()
        if not a or not b:
            return 0.0

        # Model-based similarity if available
        if self.model:
            prompt = (
                "Return only a JSON object with key 'percent' whose value is a number between 0 and 100.\n\n"
                "Task: Given STUDENT_ANSWER and ANSWER_KEY, estimate semantic similarity (0-100).\n\n"
                f"STUDENT_ANSWER:\n{a}\n\nANSWER_KEY:\n{b}\n\nRespond only with JSON like {{\"percent\": 78.5}}."
            )
            try:
                resp = self.model.generate_content(prompt)
                text = resp.text.strip()
                start = text.find("{")
                end = text.rfind("}") + 1
                json_blob = text[start:end] if start != -1 and end != -1 else text
                parsed = json.loads(json_blob)
                percent = float(parsed.get("percent", 0) or 0)
                return round(percent, 2)
            except Exception as e:
                print("similarity_percent model error:", e)
                # fallback to difflib below

        # Fallback: use difflib.SequenceMatcher
        try:
            ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            return round(ratio * 100, 2)
        except Exception as e:
            print("similarity_percent fallback error:", e)
            return 0.0

    def _simple_parse_fallback(self, pages_list):
        """Fallback parser when model is unavailable"""
        combined = "\n\n".join([p['text'] for p in pages_list])
        
        # Simple regex-based parsing
        questions = {}
        answer_map = {}
        
        # Find question patterns
        q_patterns = [
            r'Q(\d+)[\.\s]+(.*?)(?=Q\d+|$)',
            r'(\d+)[\)\.\s]+(.*?)(?=\d+\)|$)',
            r'Question\s*(\d+)[\s:]+(.*?)(?=Question\s*\d+|$)'
        ]
        
        for pattern in q_patterns:
            matches = re.finditer(pattern, combined, re.IGNORECASE | re.DOTALL)
            for match in matches:
                q_num = match.group(1)
                q_text = match.group(2).strip()[:200]  # Limit text length
                questions[q_num] = {
                    "page": 1,
                    "text": q_text,
                    "answer": None
                }
        
        return {
            "metadata": self.extract_metadata(pages_list),
            "parts": {"Part A": questions},
            "answer_map": answer_map
        }