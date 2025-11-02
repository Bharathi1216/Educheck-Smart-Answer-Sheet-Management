"""
Cleaned and fixed GeminiProcessor implementation.
- Uses 4-space indentation throughout (no tabs)
- Fixes missing except on try in key initialization loop
- Keeps multi-key parsing, quota handling, retries, rubric generation and grading
- Includes MongoDB persistence only when pymongo is available
"""
import os
import json
import re
import time
import difflib
import traceback
import concurrent.futures
from datetime import datetime
from typing import TYPE_CHECKING

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import pymongo
    HAS_PYMONGO = True
except Exception:
    pymongo = None
    HAS_PYMONGO = False

from google.api_core import exceptions as api_exceptions
from dotenv import load_dotenv

load_dotenv()


class GeminiProcessor:
    """
    Gemini client helper with:
      - multi-key support (GEMINI_API_KEYS / GEMINI_API_KEY)
      - _call_model with retries, per-attempt timeout and quota cooldown
      - parsing / rubric generation / grading helpers
      - optional MongoDB persistence when pymongo available
    """

    def __init__(self, api_key=None, model_name=None, image_model=None):
        raw_keys = api_key or os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
        self.api_keys = self._parse_api_keys(raw_keys)
        self.model_name = model_name or os.getenv("GEMINI_MODEL") or "models/gemini-pro-latest"
        self.image_model = image_model or os.getenv("GEMINI_IMAGE_MODEL") or "models/gemini-2.5-flash-image"

        # defaults that should exist even if Gemini not initialized
        self.rubrics = {
            "default": {
                "meaning_comprehension": 40,
                "key_concepts": 30,
                "technical_accuracy": 20,
                "structure": 10
            }
        }
        self.quota_exhausted_until = 0
        self.api_key = None
        self.model = None
        self.is_available = False

        # NEW: state for multi-key rotation
        self.current_key_idx = -1
        self.key_cooldowns = {}  # {key: epoch_until}
        self.bad_keys = set()    # keys that failed auth/permission permanently

        # If SDK missing, initialize object but mark unavailable
        if not genai:
            print("⚠️ google-generativeai not installed. Gemini disabled.")
            return

        if not self.api_keys:
            print("❌ No Gemini API key(s) provided. Set GEMINI_API_KEYS or GEMINI_API_KEY.")
            return

        # Try keys in order using helper
        for idx, key in enumerate(self.api_keys):
            if self._configure_model_for_key(key, idx):
                break

        if not self.model:
            print("❌ Failed to initialize Gemini with provided keys; Gemini disabled.")
            self.is_available = False

    # NEW: return current key string or None
    def _current_key(self):
        try:
            return self.api_keys[self.current_key_idx] if self.current_key_idx >= 0 else None
        except Exception:
            return None

    # NEW: configure SDK/model for a specific key index
    def _configure_model_for_key(self, key, idx):
        try:
            if key in self.bad_keys:
                return False
            # skip keys on cooldown
            if self._is_key_on_cooldown(key):
                return False
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel(self.model_name)
            self.api_key = key
            self.current_key_idx = idx
            self.is_available = True
            print(f"✅ Gemini initialized (key ending ...{str(key)[-6:]}), model={self.model_name}")
            return True
        except Exception as e:
            print(f"⚠️ Gemini init failed for key ending ...{str(key)[-6:]}: {type(e).__name__}: {e}")
            self.bad_keys.add(key)
            return False

    # NEW: check if a key is on cooldown
    def _is_key_on_cooldown(self, key):
        until = self.key_cooldowns.get(key, 0)
        return until > time.time()

    # NEW: mark a key on cooldown for seconds
    def _mark_key_cooldown(self, key, seconds):
        self.key_cooldowns[key] = time.time() + max(1, int(seconds))
        print(f"⏳ Key cooldown set for ...{str(key)[-6:]} {seconds}s")

    # NEW: rotate to next available key (not bad, not on cooldown)
    def _rotate_to_next_available_key(self, reason=""):
        if not self.api_keys:
            return False
        start = self.current_key_idx if self.current_key_idx >= 0 else -1
        n = len(self.api_keys)
        for step in range(1, n + 1):
            idx = (start + step) % n
            candidate = self.api_keys[idx]
            if candidate in self.bad_keys or self._is_key_on_cooldown(candidate):
                continue
            if self._configure_model_for_key(candidate, idx):
                if reason:
                    print(f"🔁 Switched to next Gemini key (reason: {reason}) -> ...{str(candidate)[-6:]}")
                return True
        print("❌ No alternative Gemini API keys available (all bad or on cooldown).")
        return False

    def _parse_api_keys(self, raw):
        """Parse comma-separated/list/tuple/JSON array into list of keys."""
        if not raw:
            return []
        if isinstance(raw, (list, tuple)):
            return [str(k).strip() for k in raw if k]
        s = str(raw).strip()
        # allow surrounding brackets/parentheses
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            s = s[1:-1]
        # try JSON parse if looks like JSON
        try:
            if s.startswith("[") or s.startswith("{"):
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(k).strip() for k in obj if k]
        except Exception:
            pass
        parts = [p.strip().strip('\'"') for p in s.split(",") if p.strip()]
        return parts

    def _ensure_ocr_input(self, pages_list):
        """Helper to validate OCR input."""
        if not pages_list or not isinstance(pages_list, list):
            raise ValueError("Invalid pages_list: Must be a non-empty list of page objects.")
        if not isinstance(pages_list[0], dict) or "text" not in pages_list[0]:
            raise ValueError("Invalid page object: Must be a dict with a 'text' key.")

    def _compact_prompt(self, prompt, max_len=1800):
        """Helper to truncate long prompts safely."""
        if len(prompt) > max_len:
            return prompt[:max_len] + "\n... [TRUNCATED]"
        return prompt

    def _extract_first_json(self, text):
        """Enhanced JSON extraction that handles markdown code blocks and truncated responses."""
        text = text or ""
        
        # Handle markdown-wrapped JSON (```json ... ```
        if "```json" in text:
            start_marker = text.find("```json") + 7
            end_marker = text.find("```", start_marker)
            if end_marker != -1:
                json_content = text[start_marker:end_marker].strip()
                return json_content
            else:
                # Truncated markdown - extract from ```json to end
                json_content = text[start_marker:].strip()
                return json_content
        
        # Handle markdown-wrapped without "json" (``` ... ```
        if text.count("```") >= 2:
            parts = text.split("```")
            if len(parts) >= 3:
                return parts[1].strip()
        
        # Standard JSON extraction
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return text[start:end]
        
        # Fallback: try to find incomplete JSON and fix it
        if start != -1:
            incomplete = text[start:]
            # Try to close incomplete JSON
            if incomplete.count("{") > incomplete.count("}"):
                missing_braces = incomplete.count("{") - incomplete.count("}")
                incomplete += "}" * missing_braces
            return incomplete
            
        return text

    def normalize_parsed_structure(self, parsed_data):
        """
        Normalizes a parsed structure to ensure consecutive question IDs and aligned answer maps.
        """
        if not isinstance(parsed_data, dict):
            parsed_data = {}

        # 1. Get ordered list of original labels (from 'parts' or 'ordered_questions_original')
        ordered_labels_original = []
        if "ordered_questions_original" in parsed_data and parsed_data["ordered_questions_original"]:
            ordered_labels_original = parsed_data["ordered_questions_original"]
        elif "parts" in parsed_data and parsed_data["parts"]:
            # This is a fallback to flatten the 'parts' structure if 'ordered_questions_original' is missing
            # You would need a recursive flatten function here. For simplicity, we'll assume 'ordered_questions_original' is the primary source.
            # If _flatten_questions_from_parts exists, use it.
            try:
                ordered_labels_original = self._flatten_questions_from_parts(parsed_data.get("parts", {}))
            except Exception:
                # Fallback if flatten isn't defined: just use answer_map keys
                ordered_labels_original = list(parsed_data.get("answer_map", {}).keys())
        
        if not ordered_labels_original and "answer_map" in parsed_data:
             ordered_labels_original = list(parsed_data.get("answer_map", {}).keys())

        # 2. Create consecutive mappings
        ordered_questions_consecutive = [str(i+1) for i in range(len(ordered_labels_original))]
        consec_to_label = {str(i+1): label for i, label in enumerate(ordered_labels_original)}
        label_to_consec = {label: str(i+1) for i, label in enumerate(ordered_labels_original)}

        # 3. Align the answer_map to the new consecutive keys
        answer_map_original = parsed_data.get("answer_map", {}) or {}
        answers_consecutive_aligned = {}
        for consec_key, orig_label in consec_to_label.items():
            # Find the answer for the original label
            answer = answer_map_original.get(orig_label)
            if answer is None:
                 # Try case-insensitive or partial match if direct fails (optional, but robust)
                for k, v in answer_map_original.items():
                    if str(k).strip().lower() == str(orig_label).strip().lower():
                        answer = v
                        break
            
            answers_consecutive_aligned[consec_key] = answer if answer is not None else ""

        # 4. Return the complete, normalized structure
        return {
            "metadata": parsed_data.get("metadata", {}),
            "parts": parsed_data.get("parts", {}),
            "answer_map_original": answer_map_original,
            "ordered_questions_original": ordered_labels_original,
            "ordered_questions_consecutive": ordered_questions_consecutive,
            "consec_to_label": consec_to_label,
            "label_to_consec": label_to_consec,
            "answers_consecutive_aligned": answers_consecutive_aligned
        }
    def _ensure_model(self):
        if not self.model:
            raise RuntimeError("Gemini model not initialized. Set GEMINI_API_KEY(s) and GEMINI_MODEL.")

    def is_quota_exhausted(self) -> bool:
        # Check only current key's cooldown (prefer per-key logic)
        ck = self._current_key()
        if ck and self._is_key_on_cooldown(ck):
            return True
        # fallback to legacy global flag if used elsewhere
        return getattr(self, "quota_exhausted_until", 0) > time.time()

    def _set_quota_cooldown(self, seconds: int):
        # Maintain legacy global cooldown (not used for rotation decisions)
        self.quota_exhausted_until = time.time() + int(seconds)
        print(f"Gemini quota cooldown set for {seconds}s (until {datetime.utcfromtimestamp(self.quota_exhausted_until).isoformat()}Z)")

    def _call_model(self, prompt, retries=None, backoff_seconds=None, per_attempt_timeout=None):
        """
        Call model with per-attempt timeout using ThreadPoolExecutor, retry/backoff,
        and rotate keys IMMEDIATELY on quota/permission errors.
        """
        if not self.model:
            raise RuntimeError("Gemini model not initialized")

        # defaults (drastically increased for large prompts and complex preprocessing)
        try:
            default_retries = int(os.getenv("GEMINI_RETRIES", "3"))
        except Exception:
            default_retries = 3
        try:
            default_backoff = float(os.getenv("GEMINI_BACKOFF", "2.0"))  # increased backoff
        except Exception:
            default_backoff = 2.0
        # increase default per-attempt timeout to 240s (was 180s)
        try:
            default_timeout = int(os.getenv("GEMINI_PER_ATTEMPT_TIMEOUT", "240"))
        except Exception:
            default_timeout = 240

        retries = default_retries if retries is None else int(retries)
        backoff_seconds = default_backoff if backoff_seconds is None else float(backoff_seconds)
        per_attempt_timeout = default_timeout if per_attempt_timeout is None else int(per_attempt_timeout)

        # If prompt is very large, allow an override long timeout via env var
        try:
            long_threshold = int(os.getenv("GEMINI_PROMPT_LONG_THRESHOLD", "3000"))  # lowered threshold
        except Exception:
            long_threshold = 3000
        if len(str(prompt)) > long_threshold:
            try:
                # use 360s for very long prompts
                per_attempt_timeout = int(os.getenv("GEMINI_PER_ATTEMPT_TIMEOUT_LONG", "360"))
            except Exception:
                per_attempt_timeout = 360

        last_exc = None

        def _generate():
            return self.model.generate_content(prompt)

        for attempt in range(1, max(1, retries) + 1):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_generate)
                try:
                    resp = future.result(timeout=per_attempt_timeout)
                    return resp
                except concurrent.futures.TimeoutError:
                    last_exc = TimeoutError(f"Model call timed out after {per_attempt_timeout}s (attempt {attempt})")
                    print(f"Gemini timeout (attempt {attempt}/{retries}) after {per_attempt_timeout}s")
                except Exception as e:
                    last_exc = e
                    etext = str(e)
                    print(f"Gemini call failed (attempt {attempt}/{retries}): {type(e).__name__}: {e}")

                    # Handle quota exhaustion / 429 - IMMEDIATE rotation, no retries
                    if isinstance(e, api_exceptions.ResourceExhausted) or "quota" in etext.lower() or "resourceexhausted" in etext.lower() or "429" in etext:
                        cd = 120  # shorter cooldown for faster rotation
                        m = re.search(r'Please retry in\s*(\d+)', etext, re.IGNORECASE)
                        if m:
                            cd = min(int(m.group(1)) + 5, 300)  # cap at 5 minutes
                        cur = self._current_key()
                        if cur:
                            self._mark_key_cooldown(cur, cd)
                        if self._rotate_to_next_available_key("quota"):
                            # IMMEDIATE retry with new key, don't count as attempt
                            attempt -= 1  # reset attempt counter for new key
                            continue
                        self._set_quota_cooldown(cd)
                        raise e

                    # Handle auth errors - IMMEDIATE rotation
                    if any(tok in etext.lower() for tok in ("unauthenticated", "permission", "forbidden", "401", "403")):
                        cur = self._current_key()
                        if cur:
                            self.bad_keys.add(cur)
                            print(f"🚫 Marked key ...{str(cur)[-6:]} as bad (auth/permission).")
                        if self._rotate_to_next_available_key("auth"):
                            attempt -= 1  # reset attempt counter for new key
                            continue
                        raise e

                # backoff before next attempt (only for non-rotation errors)
                if attempt < retries:
                    sleep_time = backoff_seconds * (2 ** (attempt - 1))
                    print(f"Waiting {sleep_time}s before retrying Gemini (attempt {attempt+1}/{retries})")
                    time.sleep(sleep_time)

        print("Gemini calls exhausted retries; raising last exception.")
        raise last_exc

    def analyze_json_output(self, text):
        try:
            blob = self._extract_first_json(text)
            return json.loads(blob)
        except Exception:
            try:
                return json.loads(text)
            except Exception:
                return None

    # -----------------------
    # Existing capabilities
    # -----------------------
    def resolve_mcq_ambiguity(self, text_blob, original_text=None, prefer_regex=True, question_context=None):
        if not text_blob and not original_text:
            return {}
        src = original_text or text_blob or ""
        try:
            matches = re.findall(r"(?:Q(?:uestion)?\.?\s*)?0*([0-9]{1,3})[\)\.\:\-]?\s*([A-D])\b", src, re.IGNORECASE)
            out = {f"Q{m[0]}": m[1].upper() for m in matches}
            if out or prefer_regex:
                return out
        except Exception as e:
            print("MCQ regex extraction error:", e)
        # model fallback with ultra-compact prompt
        self._ensure_model()
        prompt = f"JSON only. Map Q# to MCQ letter.\nTEXT:\n{src[:800]}\nReturn {{\"Q1\":\"A\",...}}"
        resp = self._call_model(prompt, per_attempt_timeout=60)
        text = getattr(resp, "text", "") or ""
        parsed = self.analyze_json_output(text)
        out = {}
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                out[str(k).strip()] = str(v).strip().upper() if v is not None else ""
        return out

    def extract_metadata(self, pages_list):
        combined = "\n\n".join([f"--- PAGE {p.get('page','')} ---\n{p.get('text','')}" for p in pages_list])
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
        try:
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

        # model fallback with minimal prompt
        if self.model and (not meta["course_code"] or not meta["roll"]):
            try:
                prompt = f"JSON only. Extract: course_code,total_marks,subject,date,name,roll,exam_type.\n{combined[:1200]}\nReturn {{}}"
                resp = self._call_model(prompt, per_attempt_timeout=60)
                parsed = self.analyze_json_output(getattr(resp, "text", "") or "")
                if isinstance(parsed, dict):
                    for key in meta.keys():
                        if key in parsed and parsed[key] not in (None, "", []):
                            meta[key] = parsed[key]
            except Exception:
                pass
        return meta

    def parse_pages_to_structure(self, pages_list, model_timeout=None):
        """
        Enhanced OCR parsing with complete question/answer extraction - NO CONTENT SHOULD BE MISSED.
        """
        self._ensure_ocr_input(pages_list)

        combined = "\n\n".join([f"--- PAGE {p.get('page','')} ---\n{p.get('text','')}" for p in pages_list])

        # Enhanced prompt for COMPLETE extraction - emphasize not missing anything
        prompt = (
            "CRITICAL: Extract ALL questions and answers completely. DO NOT skip or truncate any content.\n\n"
            "Parse this exam paper OCR text and return complete JSON structure:\n"
            "{\n"
            '  "metadata": {\n'
            '    "total_marks": number,\n'
            '    "course_code": "course_code",\n'
            '    "name": "student_name",\n'
            '    "roll": "roll_number",\n'
            '    "subject": "subject_name"\n'
            '  },\n'
            '  "parts": {\n'
            '    "1a": {"1.a.i": {"text": "full_question_text", "answer": "student_answer"}, "1.a.ii": {...}},\n'
            '    "2a": {"2.a.i": {...}, "2.a.ii": {...}},\n'
            '    "2b": {"2.b.i": {...}, "2.b.ii": {...}}\n'
            '  },\n'
            '  "answer_map": {\n'
            '    "1.a.i": "answer1", "1.a.ii": "answer2", "1.a.iii": "answer3",\n'
            '    "2.a.i": "answer1", "2.a.ii": "answer2", ...\n'
            '  },\n'
            '  "ordered_questions_original": ["1.a.i", "1.a.ii", "1.a.iii", "2.a.i", "2.a.ii", ...]\n'
            "}\n\n"
            "EXTRACTION RULES:\n"
            "1. EXTRACT EVERY SINGLE QUESTION - count through the text systematically\n"
            "2. PRESERVE EXACT question numbering (1.a.i, 1.a.ii, 2.a.i, 2.b.i, etc.)\n" 
            "3. For missing student answers, use empty string ''\n"
            "4. Include ALL question text completely (do not truncate)\n"
            "5. Map every question to its corresponding student answer\n"
            "6. Check you have extracted ALL questions by counting them\n\n"
            f"OCR_TEXT:\n{combined}\n\n"
            "Return ONLY the JSON - no markdown, no explanation."
        )

        use_short = os.getenv("GEMINI_USE_SHORT_PROMPTS", "1") == "1"
        # Don't truncate for parsing - we need complete extraction
        prompt_to_send = prompt if not use_short else self._compact_prompt(prompt, max_len=3500)

        self._ensure_model()
        try:
            if model_timeout is not None:
                resp = self._call_model(prompt_to_send, per_attempt_timeout=int(model_timeout))
            else:
                # Use longer timeout for complete parsing
                parse_timeout = int(os.getenv("GEMINI_PER_ATTEMPT_TIMEOUT_PARSE", "300"))
                resp = self._call_model(prompt_to_send, per_attempt_timeout=parse_timeout)

            text = getattr(resp, "text", "") or ""
            json_blob = self._extract_first_json(text)
            parsed = json.loads(json_blob)

            # Ensure proper structure and normalize answers
            if not isinstance(parsed, dict):
                parsed = {}
            
            answer_map = parsed.get("answer_map", {}) or {}
            # Normalize all answers to strings, preserve empty for missing
            for qid, ans in list(answer_map.items()):
                if ans is None:
                    answer_map[qid] = ""
                else:
                    answer_map[qid] = str(ans).strip()
            parsed["answer_map"] = answer_map
            
            # Ensure ordered_questions_original exists and is complete
            if not parsed.get("ordered_questions_original"):
                ordered = []
                if parsed.get("parts"):
                    for part_name, part_data in parsed["parts"].items():
                        if isinstance(part_data, dict):
                            # Sort question keys naturally (1.a.i before 1.a.ii)
                            sorted_keys = sorted(part_data.keys(), key=self._natural_sort_key)
                            ordered.extend(sorted_keys)
                if not ordered and answer_map:
                    ordered = sorted(answer_map.keys(), key=self._natural_sort_key)
                parsed["ordered_questions_original"] = ordered
            
            # Log extraction statistics for verification
            total_questions = len(parsed.get("ordered_questions_original", []))
            answered_questions = len([q for q, a in answer_map.items() if a.strip()])
            print(f"[parse_pages_to_structure] Extracted {total_questions} questions, {answered_questions} with answers")
            
            return parsed

        except Exception as e:
            print("Error parsing pages to structure (after retries):", type(e).__name__, str(e))
            return self._enhanced_fallback_parser(pages_list)

    def _natural_sort_key(self, question_id):
        """Sort question IDs naturally (1.a.i, 1.a.ii, 1.a.iii, 2.a.i, etc.)"""
        import re
        # Extract parts: 1.a.i -> [1, 'a', 'i']
        parts = re.findall(r'(\d+|[a-z]+)', str(question_id).lower())
        result = []
        for part in parts:
            if part.isdigit():
                result.append(int(part))
            else:
                result.append(part)
        return result

    def _enhanced_fallback_parser(self, pages_list):
        """Enhanced fallback parser that systematically extracts questions and answers"""
        combined = "\n\n".join([p.get('text', '') for p in pages_list])
        
        # Extract metadata
        metadata = self.extract_metadata(pages_list)
        
        # Enhanced question patterns - more comprehensive
        question_patterns = [
            r'(\d+)\.\s*([a-z])\s*([ivx]+|\d+)\s*[\)\.]?\s*(.*?)(?=\d+\.[a-z]\s*[ivx]+|\d+\.[a-z]\s*\d+|$)',
            r'(\d+)\.\s*([a-z])\s*[\)\.]?\s*(.*?)(?=\d+\.[a-z]|$)', 
            r'(\d+)\s*[\)\.]?\s*(.*?)(?=\d+\s*[\)\.]|$)',
            r'Question\s*(\d+)\s*[\.\:]?\s*(.*?)(?=Question\s*\d+|$)',
            r'Q(\d+)[\.\s]+(.*?)(?=Q\d+|$)'
        ]
        
        questions = {}
        answer_map = {}
        ordered_questions = []
        
        # Try each pattern to extract maximum questions
        for pattern in question_patterns:
            matches = re.finditer(pattern, combined, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match.groups()) >= 4:  # Full hierarchical match
                    q_main, q_sub, q_subsub, q_text = match.groups()[:4]
                    q_id = f"{q_main}.{q_sub}.{q_subsub}" if q_subsub else f"{q_main}.{q_sub}"
                elif len(match.groups()) >= 3:  # Partial hierarchical
                    q_main, q_sub, q_text = match.groups()[:3]
                    q_id = f"{q_main}.{q_sub}"
                else:  # Simple numbering
                    q_num, q_text = match.groups()[:2]
                    q_id = q_num
                
                if q_id not in questions:
                    questions[q_id] = {
                        "text": q_text.strip()[:500],  # Limit text length
                        "answer": ""
                    }
                    ordered_questions.append(q_id)
                    answer_map[q_id] = ""  # Default to empty
        
        # Sort questions naturally
        ordered_questions.sort(key=self._natural_sort_key)
        
        # Try to extract answers using proximity and patterns
        answer_patterns = [
            r'(?:Answer|Ans)[\s\:]*(\d+)[\s\.\)]*([^\n\r]+)',
            r'(\d+)[\s\.\)]+([A-D])\s*[\)\.]',  # MCQ answers
            r'(\d+)[\s\.\)]+(.*?)(?=\d+[\s\.\)]|$)'  # General answers
        ]
        
        for pattern in answer_patterns:
            matches = re.finditer(pattern, combined, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    q_ref, answer_text = match.groups()[:2]
                    answer_text = answer_text.strip()
                    
                    # Try to match with extracted questions
                    for q_id in ordered_questions:
                        if q_ref in q_id and answer_text:
                            answer_map[q_id] = answer_text
                            break
        
        print(f"[enhanced_fallback_parser] Extracted {len(ordered_questions)} questions with fallback method")
        
        return {
            "metadata": metadata,
            "parts": {"Part A": questions},
            "answer_map": answer_map,
            "ordered_questions_original": ordered_questions
        }

    # ...existing code...