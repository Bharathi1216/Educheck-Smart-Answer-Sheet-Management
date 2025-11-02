import re
import os
import json
from services.vision_service import VisionService
from services.gemini_service import GeminiService

class AnswerKeyProcessor:
    def __init__(self, vision_service, gemini_service):
        self.vision_service = vision_service
        self.gemini_service = gemini_service
    
    def extract_roll_numbers_from_pdf(self, pdf_path):
        """Extract roll numbers from answer key PDF"""
        try:
            # Convert PDF to images
            images = self.vision_service.extract_text_from_pdf(pdf_path)
            roll_number_mapping = {}
            
            for page_num, page_text in enumerate(images):
                # Use regex to find roll numbers in the text
                roll_numbers = self._find_roll_numbers(page_text)
                
                for roll_number in roll_numbers:
                    # Extract answers for this roll number from this page
                    answers = self._extract_answers_for_roll(page_text, roll_number)
                    if answers:
                        roll_number_mapping[roll_number] = {
                            'answers': answers,
                            'page_number': page_num + 1,
                            'raw_text': page_text
                        }
            
            return roll_number_mapping
            
        except Exception as e:
            print(f"Error extracting roll numbers from PDF: {e}")
            return {}
    
    def _find_roll_numbers(self, text):
        """Find roll numbers in text using regex patterns"""
        # Enhanced roll number patterns
        patterns = [
            r'Roll[:\s]*([A-Za-z0-9\/\-]+)',  # Roll: 12345, Roll: CS-2023/001
            r'Roll No[.\s]*([A-Za-z0-9\/\-]+)',  # Roll No. 12345
            r'Roll Number[:\s]*([A-Za-z0-9\/\-]+)',  # Roll Number: 12345
            r'ID[:\s]*([A-Za-z0-9\/\-]+)',  # ID: 12345
            r'Student ID[:\s]*([A-Za-z0-9\/\-]+)',  # Student ID: 12345
            r'Registration No[:\s]*([A-Za-z0-9\/\-]+)',  # Registration No: 12345
            r'^\s*([A-Za-z0-9\/\-]{5,15})\s*$',  # Standalone roll numbers
        ]
        
        found_roll_numbers = set()
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # Take first group if it's a tuple
                roll_num = match.strip()
                if len(roll_num) >= 3:  # Valid roll numbers should be at least 3 chars
                    found_roll_numbers.add(roll_num)
        
        return list(found_roll_numbers)
    
    def _extract_answers_for_roll(self, text, roll_number):
        """Extract answers for a specific roll number from text"""
        try:
            # Use AI to identify and extract answers with strict question number preservation
            prompt = f"""
            Extract the student's answers from the following text for roll number {roll_number}.
            CRITICAL RULES:
            1. PRESERVE ORIGINAL QUESTION NUMBERS EXACTLY AS WRITTEN - DO NOT RENUMBER
            2. For missing answers, use empty string ""
            3. IGNORE completely any struck-out/crossed-out answers
            4. For 'write any two' type questions: include ALL correct answers
            
            Text: {text}
            
            Return the answers in this exact JSON format:
            {{
                "answers": [
                    {{
                        "question_number": "1",  // PRESERVE ORIGINAL FORMAT
                        "answer_text": "extracted answer"
                    }},
                    {{
                        "question_number": "2a",  // PRESERVE SUBPARTS
                        "answer_text": "extracted answer" 
                    }}
                ]
            }}
            
            If you cannot find answers, return empty array.
            Return only the JSON, no additional text.
            """
            
            response = self.gemini_service.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            
            result = json.loads(response_text)
            
            # Ensure empty strings for missing answers
            for answer in result.get('answers', []):
                if answer.get('answer_text') is None:
                    answer['answer_text'] = ""
            
            return result.get('answers', [])
            
        except Exception as e:
            print(f"Error extracting answers for roll {roll_number}: {e}")
            return []
    
    def process_answer_key_pdf(self, pdf_path, assignment_id, assignment_model):
        """Process answer key PDF and store in database"""
        try:
            # Extract roll numbers and answers
            roll_answers_mapping = self.extract_roll_numbers_from_pdf(pdf_path)
            
            # Store in database
            for roll_number, answer_data in roll_answers_mapping.items():
                assignment_model.update_answer_key(
                    assignment_id, 
                    roll_number, 
                    answer_data['answers']
                )
            
            return {
                "processed_roll_numbers": list(roll_answers_mapping.keys()),
                "total_processed": len(roll_answers_mapping),
                "status": "success"
            }
            
        except Exception as e:
            print(f"Error processing answer key PDF: {e}")
            return {
                "processed_roll_numbers": [],
                "total_processed": 0,
                "status": "error",
                "error": str(e)
            }