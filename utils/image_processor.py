import os
import io
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image, ImageEnhance, ImageFilter
import pdf2image
import tempfile
import time
from utils.gemini_processor import GeminiProcessor
from google.oauth2 import service_account
from google.api_core import exceptions as api_exceptions
import base64
import requests  # <-- added

class ImageProcessor:
    """
    Essential image processing for answer sheet evaluation
    """
    
    def __init__(self, google_vision_key_path=None, gemini_api_key=None):
        # Resolve credential path: explicit argument -> env -> common filenames
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        candidates = []
        if google_vision_key_path:
            candidates.append(google_vision_key_path)
        env_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or os.getenv('GOOGLE_VISION_KEY_PATH')
        if env_path:
            candidates.append(env_path)
        candidates.extend([
            os.path.join(project_root, 'service-account-key.json'),
            os.path.join(project_root, 'google-vision-key.json'),
            os.path.join(project_root, 'service-account.json')
        ])

        resolved = None
        for c in candidates:
            if not c:
                continue
            c_abs = c if os.path.isabs(c) else os.path.abspath(os.path.join(project_root, c))
            if os.path.exists(c_abs):
                resolved = c_abs
                break

        if resolved:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = resolved
            print(f"[ImageProcessor] Using Google Vision credentials: {resolved}")
        else:
            print("[ImageProcessor] ERROR: No Google Vision service-account JSON found. Set GOOGLE_APPLICATION_CREDENTIALS to a valid path.")
            # Do not continue without service-account
            self.vision_client = None
            raise RuntimeError("Google Vision service-account JSON not found. Set GOOGLE_APPLICATION_CREDENTIALS to a valid JSON.")

        # Create vision client via service-account credentials
        try:
            creds = service_account.Credentials.from_service_account_file(resolved)
            self.vision_client = vision.ImageAnnotatorClient(credentials=creds)
            print(f"[ImageProcessor] Vision client initialized successfully.")
        except Exception as e:
            print(f"[ImageProcessor] Failed to initialize Vision client from {resolved}: {e}")
            self.vision_client = None
            raise

        self.gemini_processor = GeminiProcessor(gemini_api_key) if gemini_api_key else None
    
    def _vision_rest_annotate_image_bytes(self, content_bytes, api_key, feature="TEXT_DETECTION"):
        # Remove REST path — we do not use API key fallback in this setup
        raise RuntimeError("REST Vision API usage is disabled. Provide a valid service-account JSON and set GOOGLE_APPLICATION_CREDENTIALS.")

    def extract_text_with_vision_api(self, image_path):
        """
        Extract text using the Vision service-account client only.
        """
        if not self.vision_client:
            raise RuntimeError("Vision client not initialized. Set GOOGLE_APPLICATION_CREDENTIALS to a valid service-account JSON.")
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = self.vision_client.text_detection(image=image)
            if getattr(response, "error", None) and getattr(response.error, "message", None):
                raise Exception(f"Google Vision API error: {response.error.message}")
            texts = response.text_annotations
            full = texts[0].description if texts else ""

            result = {
                'full_text': full,
                'text_blocks': [],
                'success': True
            }

            if texts:
                # When REST path, text annotations are dicts rather than objects
                word_details = []
                for ta in (texts[1:] if len(texts) > 1 else []):
                    # REST uses 'description' and 'boundingPoly' keys
                    if isinstance(ta, dict):
                        desc = ta.get('description', '')
                        verts = ta.get('boundingPoly', {}).get('vertices', [])
                        bounding_box = [(v.get('x', 0), v.get('y', 0)) for v in verts]
                        word_details.append({'text': desc, 'bounding_box': bounding_box})
                    else:
                        # gcloud object case
                        word_details.append({
                            'text': ta.description,
                            'bounding_box': [(vertex.x, vertex.y) for vertex in ta.bounding_poly.vertices]
                        })

                result['text_blocks'] = self._group_words_into_blocks(word_details)
                print(f"Extracted text with {len(result['text_blocks'])} organized blocks")

            return result

        except Exception as e:
            print(f"Error in text extraction: {e}")
            return {'full_text': '', 'text_blocks': [], 'success': False, 'error': str(e)}
    
    def _group_words_into_blocks(self, word_details, vertical_threshold=20):
        """
        ESSENTIAL: Group words into logical lines using spatial analysis
        Critical for separating different answers and maintaining structure
        """
        if not word_details:
            return []
        
        blocks = []
        current_block = [word_details[0]]
        
        for i in range(1, len(word_details)):
            current_word = word_details[i]
            prev_word = word_details[i-1]
            
            # Spatial analysis: Check if words are on same line
            current_y = self._get_average_y(current_word['bounding_box'])
            prev_y = self._get_average_y(prev_word['bounding_box'])
            
            if abs(current_y - prev_y) <= vertical_threshold:
                current_block.append(current_word)
            else:
                # New line detected - create new block
                if current_block:
                    blocks.append(self._create_text_block(current_block))
                current_block = [current_word]
        
        if current_block:
            blocks.append(self._create_text_block(current_block))
        
        return blocks
    
    def _get_average_y(self, bounding_box):
        """
        ESSENTIAL: Calculate Y position for spatial analysis
        """
        if not bounding_box:
            return 0
        y_coords = [vertex[1] for vertex in bounding_box]
        return sum(y_coords) / len(y_coords)
    
    def _create_text_block(self, words):
        """
        ESSENTIAL: Create organized text blocks from grouped words
        Maintains answer structure and readability
        """
        block_text = ' '.join(word['text'] for word in words)
        
        return {
            'text': block_text,
            'bounding_box': self._calculate_block_bounds(words),  # Essential for layout understanding
            'word_count': len(words)
        }
    
    def _calculate_block_bounds(self, words):
        """
        ESSENTIAL: Bounding box for spatial understanding
        Important for answer region identification
        """
        if not words or not words[0]['bounding_box']:
            return []
        
        all_x = []
        all_y = []
        for word in words:
            for vertex in word['bounding_box']:
                all_x.append(vertex[0])
                all_y.append(vertex[1])
        
        return [
            (min(all_x), min(all_y)),
            (max(all_x), min(all_y)),
            (max(all_x), max(all_y)),
            (min(all_x), max(all_y))
        ]
    
    def extract_text_from_pdf_vision(self, pdf_path, dpi=250):
        """
        Process PDF answer sheets with spatial analysis
        """
        try:
            print(f"Processing PDF: {pdf_path}")
            
            images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
            print(f"Converted {len(images)} pages")
            
            extraction_results = []
            
            for i, image in enumerate(images):
                print(f"Processing page {i+1}")
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_image_path = temp_file.name
                    image.save(temp_image_path, 'PNG', quality=95)
                
                try:
                    page_result = self.extract_text_with_vision_api(temp_image_path)
                    page_result['page_number'] = i + 1
                    extraction_results.append(page_result)
                    
                finally:
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
            
            return extraction_results
            
        except Exception as e:
            print(f"Error in PDF processing: {e}")
            return []
    
    def extract_and_correct_handwriting(self, image_path):
        """
        Complete pipeline: Extract → Correct → Provide feedback
        """
        try:
            # Extract text with spatial analysis
            vision_result = self.extract_text_with_vision_api(image_path)
            
            if not vision_result['success']:
                return {
                    'original_text': '',
                    'corrected_text': '',
                    'text_blocks': [],
                    'success': False
                }
            
            original_text = vision_result['full_text']
            
            # Correct handwriting using Gemini
            if original_text.strip() and self.gemini_processor:
                corrected_text = self.gemini_processor.correct_handwritten_text(original_text, "auto")
            else:
                corrected_text = original_text
            
            return {
                'original_text': original_text,
                'corrected_text': corrected_text,
                'text_blocks': vision_result['text_blocks'],  # Essential spatial data
                'success': True
            }
            
        except Exception as e:
            print(f"Error in processing: {e}")
            return {
                'original_text': '',
                'corrected_text': '',
                'text_blocks': [],
                'success': False,
                'error': str(e)
            }
    
    def process_answer_sheet_pdf(self, pdf_path, dpi=250):
        """
        Process complete answer sheet with essential spatial analysis
        """
        try:
            extraction_results = self.extract_text_from_pdf_vision(pdf_path, dpi=dpi)
            
            processed_pages = []
            for page_result in extraction_results:
                original_text = page_result.get('full_text', '')
                
                if original_text.strip() and self.gemini_processor:
                    corrected_text = self.gemini_processor.correct_handwritten_text(original_text, "auto")
                else:
                    corrected_text = original_text
                
                processed_pages.append({
                    'page_number': page_result.get('page_number', 0),
                    'original_text': original_text,
                    'corrected_text': corrected_text,
                    'text_blocks': page_result.get('text_blocks', []),  # Keep spatial data
                    'has_content': bool(original_text.strip())
                })
            
            return processed_pages
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []
    
    def get_answers_by_region(self, vision_result, y_start=0, y_end=float('inf')):
        """
        ESSENTIAL: Extract answers from specific regions using spatial data
        Useful for answer box detection and organization
        """
        blocks_in_region = []
        for block in vision_result.get('text_blocks', []):
            if block['bounding_box']:
                block_y = self._get_average_y(block['bounding_box'])
                if y_start <= block_y <= y_end:
                    blocks_in_region.append(block)
        
        return blocks_in_region