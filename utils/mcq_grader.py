import re
import json
from typing import Dict, List, Any

class MCQGrader:
    """
    Handles Multiple Choice Question evaluation for answer sheets
    - Objective answer matching with flexible patterns
    - OMR-style answer sheet processing
    - Partial credit calculation
    - Integration with Gemini for ambiguous cases
    """
    
    def __init__(self, gemini_processor=None):
        self.mcq_patterns = {
            'single_choice': r'^[A-D]$',
            'multiple_choice': r'^[A-D]+$',
            'true_false': r'^[TF]$',
            'numeric': r'^\d+$'
        }
        self.gemini_processor = gemini_processor
    
    def extract_mcq_answers_from_text(self, text, num_questions=10):
        """
        Extract MCQ answers from OCR text with various formats
        Handles: Q1. A, 1) B, Question 1: C, etc.
        """
        try:
            answers = {}
            
            # Enhanced patterns for different question formats
            patterns = [
                r'Q(\d+)[\.\s]*([A-D])',                           # Q1. A, Q2 B
                r'(\d+)[\)\.\s]*([A-D])',                         # 1) A, 2. B
                r'Question\s*(\d+)[\s:]*([A-D])',                 # Question 1: A
                r'(\d+)\s*\.\s*([A-D])',                          # 1. A
                r'Ans\s*(\d+)[\s:]*([A-D])',                      # Ans 1: A
                r'Answer\s*(\d+)[\s:]*([A-D])',                   # Answer 1: A
                r'\((\d+)\)\s*([A-D])',                           # (1) A
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    q_num, answer = match
                    q_num = int(q_num)
                    if 1 <= q_num <= num_questions:
                        answers[q_num] = answer.upper()
            
            # Also look for answer grids/patterns
            grid_answers = self._extract_grid_answers(text, num_questions)
            answers.update(grid_answers)
            
            return answers
            
        except Exception as e:
            print(f"Error extracting MCQ answers: {e}")
            return {}
    
    def _extract_grid_answers(self, text, num_questions):
        """
        Extract answers from OMR-style grids
        Example: 
          1. ☑ A ☐ B ☐ C ☐ D
          2. ☐ A ☐ B ☑ C ☐ D
        """
        try:
            answers = {}
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Pattern for numbered questions with options
                grid_pattern = r'(\d+)\.\s*(?:[☑☐□■▢▣◼◻]?\s*[A-D]\s*)+'
                match = re.search(grid_pattern, line)
                if match:
                    q_num = int(match.group(1))
                    if 1 <= q_num <= num_questions:
                        # Find which option is selected
                        if '☑' in line or '■' in line or '◼' in line:
                            for option in ['A', 'B', 'C', 'D']:
                                option_pattern = f'[☑■◼]\\s*{option}'
                                if re.search(option_pattern, line):
                                    answers[q_num] = option
                                    break
            
            return answers
            
        except Exception as e:
            print(f"Error extracting grid answers: {e}")
            return {}
    
    def grade_mcq_answer(self, student_answer, correct_answer, question_type='single_choice', negative_marking=False):
        """
        Grade MCQ answers with flexible matching and partial credit
        """
        try:
            # Normalize answers
            student_answer = str(student_answer).upper().strip()
            correct_answer = str(correct_answer).upper().strip()
            
            # Handle empty answers
            if not student_answer:
                return self._get_mcq_result(0, False, "No answer provided", student_answer, correct_answer)
            
            # Validate answer format
            if not self._validate_mcq_format(student_answer, question_type):
                return self._get_mcq_result(0, False, "Invalid answer format", student_answer, correct_answer)
            
            # Grade based on question type
            if question_type == 'single_choice':
                return self._grade_single_choice(student_answer, correct_answer, negative_marking)
            elif question_type == 'multiple_choice':
                return self._grade_multiple_choice(student_answer, correct_answer, negative_marking)
            elif question_type == 'true_false':
                return self._grade_true_false(student_answer, correct_answer, negative_marking)
            elif question_type == 'numeric':
                return self._grade_numeric(student_answer, correct_answer)
            else:
                return self._grade_generic_mcq(student_answer, correct_answer)
                
        except Exception as e:
            print(f"Error grading MCQ: {e}")
            return self._get_mcq_result(0, False, "Grading error occurred", student_answer, correct_answer)
    
    def _grade_single_choice(self, student_answer, correct_answer, negative_marking=False):
        """Grade single choice questions"""
        is_correct = student_answer == correct_answer
        
        if is_correct:
            score = 1
            feedback = "Correct!"
        else:
            score = -1 if negative_marking else 0
            feedback = f"Correct answer is {correct_answer}"
        
        return self._get_mcq_result(score, is_correct, feedback, student_answer, correct_answer)
    
    def _grade_multiple_choice(self, student_answer, correct_answer, negative_marking=False):
        """Grade multiple choice questions with partial credit"""
        student_choices = set(student_answer)
        correct_choices = set(correct_answer)
        
        correct_selected = len(student_choices.intersection(correct_choices))
        incorrect_selected = len(student_choices - correct_choices)
        total_correct = len(correct_choices)
        
        if negative_marking:
            # Penalty for wrong answers
            score = max(0, (correct_selected - incorrect_selected) / total_correct)
        else:
            # Only reward correct answers
            score = correct_selected / total_correct
        
        is_correct = score > 0.7  # Threshold for "correct"
        
        feedback = f"Selected {correct_selected}/{total_correct} correct options"
        if incorrect_selected > 0:
            feedback += f", {incorrect_selected} incorrect options"
        
        return self._get_mcq_result(score, is_correct, feedback, student_answer, correct_answer)
    
    def _grade_true_false(self, student_answer, correct_answer, negative_marking=False):
        """Grade True/False questions"""
        is_correct = student_answer == correct_answer
        
        if is_correct:
            score = 1
            feedback = "Correct!"
        else:
            score = -1 if negative_marking else 0
            feedback = f"Correct answer is {correct_answer}"
        
        return self._get_mcq_result(score, is_correct, feedback, student_answer, correct_answer)
    
    def _grade_numeric(self, student_answer, correct_answer):
        """Grade numeric answers (for questions like 'Select the correct number')"""
        try:
            # Remove any non-numeric characters and compare
            student_num = re.sub(r'[^\d]', '', student_answer)
            correct_num = re.sub(r'[^\d]', '', correct_answer)
            
            is_correct = student_num == correct_num
            score = 1 if is_correct else 0
            
            feedback = "Correct!" if is_correct else f"Correct answer is {correct_answer}"
            
            return self._get_mcq_result(score, is_correct, feedback, student_answer, correct_answer)
        except:
            return self._get_mcq_result(0, False, "Invalid numeric format", student_answer, correct_answer)
    
    def _grade_generic_mcq(self, student_answer, correct_answer):
        """Generic MCQ grading with flexible matching"""
        is_correct = self._compare_answers(student_answer, correct_answer)
        score = 1 if is_correct else 0
        
        feedback = "Correct!" if is_correct else f"Expected: {correct_answer}"
        
        return self._get_mcq_result(score, is_correct, feedback, student_answer, correct_answer)
    
    def calculate_mcq_score(self, student_answers, correct_answers, scoring_scheme='standard', negative_marking=False, total_marks: float = None):
        """
        Calculate total MCQ score for an entire answer sheet.

        - total_marks: if provided, distribute these marks evenly across questions and return totals/percentage based on it.
                       If None, fallback to 1 mark per question.
        - scoring_scheme kept for backward compatibility (e.g., 'negative_marking').
        """
        try:
            total_questions = len(correct_answers)
            if total_questions == 0:
                return {"total_score": 0, "correct_count": 0, "total_questions": 0, "percentage": 0, "detailed_results": {}}
            
            correct_count = 0
            total_obtained_marks = 0.0
            detailed_results = {}

            # Determine per-question mark from total_marks if provided
            if total_marks is not None and total_questions > 0:
                per_q_mark = float(total_marks) / float(total_questions)
                max_total_marks = float(total_marks)
            else:
                per_q_mark = 1.0
                max_total_marks = float(per_q_mark * total_questions)
            
            for q_num, correct_answer in correct_answers.items():
                student_answer = student_answers.get(q_num, '')
                question_type = self._detect_question_type(correct_answer)
                
                result = self.grade_mcq_answer(
                    student_answer, 
                    correct_answer, 
                    question_type, 
                    negative_marking
                )
                
                # Convert grader 'score' to awarded marks
                raw_score = result.get('score', 0)
                try:
                    raw_score = float(raw_score)
                except Exception:
                    raw_score = 0.0

                if negative_marking:
                    awarded_marks = round(raw_score * per_q_mark, 2)
                else:
                    awarded_marks = round(max(0.0, raw_score) * per_q_mark, 2)

                # update counters
                if result.get('is_correct'):
                    correct_count += 1
                total_obtained_marks += awarded_marks

                # attach awarded marks into detailed result
                result_with_marks = dict(result)
                result_with_marks['awarded_marks'] = awarded_marks
                result_with_marks['max_marks'] = per_q_mark

                detailed_results[q_num] = result_with_marks
            
            # Percentage is based on marks if total_marks provided; else based on questions (as earlier)
            if max_total_marks > 0:
                percentage = (total_obtained_marks / max_total_marks) * 100
            else:
                percentage = 0.0

            return {
                "total_score": round(total_obtained_marks, 2),
                "correct_count": correct_count,
                "total_questions": total_questions,
                "percentage": round(percentage, 2),
                "per_question_mark": round(per_q_mark, 2),
                "total_marks": round(max_total_marks, 2),
                "detailed_results": detailed_results
            }
            
        except Exception as e:
            print(f"Error calculating MCQ score: {e}")
            return {"total_score": 0, "correct_count": 0, "total_questions": 0, "percentage": 0, "detailed_results": {}}
    
    def resolve_ambiguous_answers(self, extracted_text, correct_answers):
        """
        Use Gemini to resolve ambiguous or poorly extracted MCQ answers
        """
        if not self.gemini_processor:
            return extracted_text
        
        try:
            prompt = f"""
            Analyze this extracted MCQ answers and resolve any ambiguities:
            
            EXTRACTED TEXT: {extracted_text}
            EXPECTED QUESTIONS: {list(correct_answers.keys())}
            
            Return a JSON with cleaned answers in format:
            {{
                "cleaned_answers": {{
                    "1": "A",
                    "2": "B",
                    ...
                }},
                "confidence": 0.8,
                "ambiguous_questions": [3, 5]  # list of question numbers that were unclear
            }}
            
            Only return the JSON object.
            """
            
            response = self.gemini_processor.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            
            result = json.loads(response_text)
            return result.get('cleaned_answers', {})
            
        except Exception as e:
            print(f"Error resolving ambiguous answers with Gemini: {e}")
            return extracted_text
    
    def _validate_mcq_format(self, answer, question_type):
        """Validate MCQ answer format"""
        pattern = self.mcq_patterns.get(question_type, r'^[A-Z0-9]+$')
        return bool(re.match(pattern, answer))
    
    def _compare_answers(self, answer1, answer2):
        """Flexible answer comparison"""
        return answer1.strip().upper() == answer2.strip().upper()
    
    def _detect_question_type(self, correct_answer):
        """Detect question type based on correct answer format"""
        correct_answer = str(correct_answer).upper().strip()
        
        if len(correct_answer) == 1 and correct_answer in ['T', 'F']:
            return 'true_false'
        elif len(correct_answer) == 1 and correct_answer in ['A', 'B', 'C', 'D']:
            return 'single_choice'
        elif len(correct_answer) > 1 and all(c in 'ABCD' for c in correct_answer):
            return 'multiple_choice'
        elif correct_answer.isdigit():
            return 'numeric'
        else:
            return 'single_choice'  # Default fallback
    
    def _get_mcq_result(self, score, is_correct, feedback, student_answer, correct_answer):
        """Standardized MCQ result format"""
        return {
            "score": round(score, 2),
            "is_correct": is_correct,
            "feedback": feedback,
            "student_answer": student_answer,
            "correct_answer": correct_answer
        }