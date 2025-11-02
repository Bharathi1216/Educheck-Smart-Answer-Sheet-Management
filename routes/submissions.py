from flask import Blueprint, request, jsonify
from services.storage_service import StorageService
from models.submission import SubmissionModel
from models.assignment import AssignmentModel
from models.student import StudentModel
from models.user import UserModel
from utils.image_processor import ImageProcessor
from utils.gemini_processor import GeminiProcessor
from config.database import db_instance
from config.settings import settings  # <-- added
import os
import shutil  # <-- added
from datetime import datetime

submissions_bp = Blueprint('submissions', __name__)

# Initialize services
db = db_instance
submission_model = SubmissionModel(db)
assignment_model = AssignmentModel(db)
student_model = StudentModel(db)
user_model = UserModel(db)
storage_service = StorageService(os.getenv('STORAGE_TYPE', 'local'))

# Initialize processors
image_processor = ImageProcessor(
    google_vision_key_path="google-vision-key.json",
    gemini_api_key=os.getenv('GEMINI_API_KEY')
)
gemini_processor = GeminiProcessor(os.getenv('GEMINI_API_KEY'))

@submissions_bp.route('/api/submit/answer', methods=['POST'])
def submit_answer():
    """
    Both Students and Teachers can submit answer sheets
    - Students: Submit their answer sheets for evaluation
    - Teachers: Submit answer keys for assignments
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        user_id = request.form.get('user_id')
        user_role = request.form.get('user_role')  # 'student' or 'teacher'
        assignment_id = request.form.get('assignment_id')
        # FIX: use singular 'student_answer' consistent across app
        submission_type = request.form.get('submission_type', 'student_answer')  # 'student_answer' or 'teacher_answer_key'
        
        if not user_id or not user_role:
            return jsonify({"error": "Missing user_id or user_role"}), 400
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Verify user exists
        user = user_model.find_by_id(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Validate user role matches submission type
        if submission_type == 'student_answer' and user_role != 'student':
            return jsonify({"error": "Only students can submit answer sheets"}), 400
        
        if submission_type == 'teacher_answer_key' and user_role != 'teacher':
            return jsonify({"error": "Only teachers can submit answer keys"}), 400
        
        # For student submissions, verify assignment exists and allows student uploads
        if submission_type == 'student_answer' and assignment_id:
            assignment = assignment_model.get_by_id(assignment_id)
            if not assignment:
                return jsonify({"error": "Assignment not found"}), 404
            
            # ✅ NEW CHECK: Verify assignment allows student uploads
            if assignment.upload_mode != 'student':
                return jsonify({"error": "This assignment does not allow student uploads"}), 400
        
        # For teacher answer keys, assignment_id is optional
        if submission_type == 'teacher_answer_key' and assignment_id:
            assignment = assignment_model.get_by_id(assignment_id)
            if not assignment:
                return jsonify({"error": "Assignment not found"}), 404
        
        # Get additional user data
        roll_number = None
        if user_role == 'student':
            student = student_model.get_by_user_id(user_id)
            if student:
                roll_number = student.roll_number
        
        # Save uploaded file
        folder = 'submissions' if submission_type == 'student_answer' else 'answer_keys'
        file_path = storage_service.save_file(file, folder)

        # Normalize saved path and ensure it's under the uploads dir used by processor
        uploads_dir = os.path.abspath(getattr(settings, "UPLOAD_FOLDER", "uploads") or "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.abspath(file_path)

        # If saved file is not under uploads_dir, copy it there and update file_path
        if not os.path.commonpath([uploads_dir, file_path]) == uploads_dir:
            dest_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{os.path.basename(file_path)}"
            dest_path = os.path.join(uploads_dir, dest_name)
            try:
                shutil.copy2(file_path, dest_path)
                print(f"[submissions] Copied saved file to uploads folder for processing: {dest_path}")
                file_path = dest_path
            except Exception as e:
                print(f"[submissions] Could not copy file into uploads folder: {e}")
                # fallback: continue with original file_path but log warning
                print(f"[submissions] Proceeding with original saved path: {file_path}")
        else:
            print(f"[submissions] Saved file path (in uploads folder): {file_path}")

        # Prepare extracted_texts variable for DB
        extracted_texts_list = []
        corrected_texts = []
        text_blocks = []

        # Process the uploaded file
        if file_path.endswith('.pdf'):
            results = image_processor.process_answer_sheet_pdf(file_path)
            corrected_texts = [page['corrected_text'] for page in results if page.get('corrected_text')]
            # collect original/extracted texts per page
            extracted_texts_list = [page.get('original_text', '') for page in results]
            for page in results:
                text_blocks.extend(page.get('text_blocks', []))
            
            # For PDFs, we might have multiple pages of answers
            if submission_type == 'teacher_answer_key':
                # For teacher answer keys, extract roll numbers and organize answers
                roll_answer_mapping = {}
                for i, page in enumerate(results):
                    if page.get('corrected_text'):
                        # Simple extraction - you can enhance this with roll number detection
                        roll_answer_mapping[f"page_{i+1}"] = [page['corrected_text']]
                answers_data = roll_answer_mapping
            else:
                answers_data = corrected_texts
                
        else:
            # Single image file
            result = image_processor.extract_and_correct_handwriting(file_path)
            corrected_texts = [result['corrected_text']] if result.get('corrected_text') else []
            text_blocks = result.get('text_blocks', [])
            answers_data = corrected_texts
            extracted_texts_list = [result.get('original_text', '')] if result.get('original_text') else []
        
        # Create submission record
        submission_data = {
            'submission_type': submission_type,
            'user_id': user_id,
            'user_role': user_role,
            'student_id': user_id if user_role == 'student' else None,
            'teacher_id': user_id if user_role == 'teacher' else None,
            'roll_number': roll_number,
            'assignment_id': assignment_id if assignment_id else None,
            'answers': answers_data,
            'extracted_text': extracted_texts_list,            # <-- use unified var
            'corrected_text': corrected_texts,
            'text_blocks': text_blocks,
            'original_filename': file.filename,
            'file_path': file_path,
            'status': 'submitted',
            'submitted_at': datetime.utcnow()
        }
        
        submission_id = submission_model.create_submission(submission_data)
        
        # Remove local file when using local storage to avoid keeping PDFs/images
        try:
            if os.getenv('STORAGE_TYPE', 'local') == 'local' and os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

        # If teacher submitted answer key, update assignment
        if submission_type == 'teacher_answer_key' and assignment_id:
            # Store the answer key in assignment for future evaluations
            assignment_model.update_answer_key(
                assignment_id, 
                "teacher_key",  # You can enhance this with roll number mapping
                corrected_texts
            )
        
        response_data = {
            "message": "File processed successfully",
            "submission_id": submission_id,
            "submission_type": submission_type,
            "corrected_text": corrected_texts[0] if corrected_texts else "",
            "status": "submitted"
        }
        
        if submission_type == 'teacher_answer_key':
            response_data["message"] = "Answer key uploaded successfully"
        else:
            response_data["message"] = "Answer sheet submitted successfully"
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@submissions_bp.route('/api/assignments/<assignment_id>/bulk-upload-answers', methods=['POST'])
def bulk_upload_student_answers(assignment_id):
    """Teacher uploads answers for multiple students (for exam mode)"""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        teacher_id = request.form.get('teacher_id')
        roll_numbers = request.form.getlist('roll_numbers')  # Array of roll numbers
        
        if not teacher_id:
            return jsonify({"error": "Teacher ID required"}), 400
        
        # Verify assignment exists and is in teacher upload mode
        assignment = assignment_model.get_by_id(assignment_id)
        if not assignment:
            return jsonify({"error": "Assignment not found"}), 404
        
        if assignment.upload_mode != 'teacher':
            return jsonify({"error": "This assignment is not in teacher upload mode"}), 400
        
        if len(files) != len(roll_numbers):
            return jsonify({"error": "Number of files must match number of roll numbers"}), 400
        
        results = []
        
        for i, file in enumerate(files):
            if file.filename == '':
                continue
                
            roll_number = roll_numbers[i]
            
            # Find student by roll number
            student = student_model.get_by_roll_number(roll_number)
            if not student:
                results.append({
                    'roll_number': roll_number,
                    'status': 'error',
                    'error': 'Student not found'
                })
                continue
            
            # Save file
            file_path = storage_service.save_file(file, 'bulk_answers')
            
            # Process answer sheet
            if file_path.endswith('.pdf'):
                pdf_results = image_processor.process_answer_sheet_pdf(file_path)
                corrected_texts = [page['corrected_text'] for page in pdf_results if page.get('corrected_text')]
                text_blocks = []
                for page in pdf_results:
                    text_blocks.extend(page.get('text_blocks', []))
            else:
                result = image_processor.extract_and_correct_handwriting(file_path)
                corrected_texts = [result['corrected_text']] if result['corrected_text'] else []
                text_blocks = result.get('text_blocks', [])
            
            # Create submission as if student submitted it
            submission_data = {
                'submission_type': 'student_answer',
                'user_id': student.user_id,  # Student's user ID
                'user_role': 'student',
                'student_id': student.user_id,
                'roll_number': roll_number,
                'assignment_id': assignment_id,
                'answers': corrected_texts,
                'corrected_text': corrected_texts,
                'text_blocks': text_blocks,
                'original_filename': file.filename,
                'file_path': file_path,
                'status': 'submitted',
                'uploaded_by_teacher': True,  # Flag to indicate teacher uploaded
                'teacher_uploader_id': teacher_id,
                'submitted_at': datetime.utcnow()
            }
            
            submission_id = submission_model.create_submission(submission_data)
            # Remove local file when using local storage to avoid keeping PDFs/images
            try:
                if os.getenv('STORAGE_TYPE', 'local') == 'local' and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
            
            results.append({
                'roll_number': roll_number,
                'student_name': student.name,
                'submission_id': submission_id,
                'status': 'success',
                'corrected_text': corrected_texts[0] if corrected_texts else ""
            })
        
        return jsonify({
            "message": f"Processed {len(results)} student answers",
            "assignment_id": assignment_id,
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@submissions_bp.route('/api/submissions/<submission_id>/evaluate', methods=['POST'])
def evaluate_submission(submission_id):
    """Evaluate student submission against answer key"""
    try:
        # Get submission
        submission = submission_model.get_by_id(submission_id)
        if not submission:
            return jsonify({"error": "Submission not found"}), 404
        
        # Only evaluate student answers, not teacher answer keys
        if submission.submission_type != 'student_answer':
            return jsonify({"error": "Only student answers can be evaluated"}), 400
        
        # Get assignment and answer key
        assignment = assignment_model.get_by_id(submission.assignment_id)
        if not assignment:
            return jsonify({"error": "Assignment not found"}), 404
        
        # Get teacher's answer key
        answer_key_text = ""
        if assignment.answer_key:
            # Try to get answer key for this student's roll number
            if submission.roll_number and submission.roll_number in assignment.answer_key:
                answer_key_text = assignment.answer_key[submission.roll_number][0]
            elif 'teacher_key' in assignment.answer_key:
                answer_key_text = assignment.answer_key['teacher_key'][0]
            elif len(assignment.answer_key) > 0:
                # Get first available answer key
                first_key = next(iter(assignment.answer_key.values()))
                answer_key_text = first_key[0] if first_key else ""
        
        student_answer = submission.corrected_text[0] if submission.corrected_text else ""
        
        if not student_answer:
            return jsonify({"error": "No student answer to evaluate"}), 400
        
        if not answer_key_text:
            return jsonify({"error": "No answer key found for evaluation"}), 400
        
        # Evaluate using Gemini
        evaluation = gemini_processor.evaluate_answer(
            student_answer=student_answer,
            correct_answer=answer_key_text,
            rubric=assignment.rubric
        )
        
        # Update submission with evaluation results
        submission_model.add_evaluation_results(
            submission_id,
            scores=evaluation,
            feedback=[evaluation.get('feedback', 'No feedback available')],
            total_score=evaluation.get('overall_score', 0)
        )
        
        return jsonify({
            "message": "Evaluation completed successfully",
            "submission_id": submission_id,
            "overall_score": evaluation.get('overall_score', 0),
            "scores": evaluation,
            "status": "evaluated"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@submissions_bp.route('/api/submissions/batch-evaluate', methods=['POST'])
def batch_evaluate_submissions():
    """Evaluate all pending submissions for an assignment"""
    try:
        data = request.json
        assignment_id = data.get('assignment_id')
        teacher_id = data.get('teacher_id')
        
        if not assignment_id:
            return jsonify({"error": "Assignment ID is required"}), 400
        
        # Get all pending submissions for this assignment
        pending_submissions = submission_model.get_by_assignment(assignment_id, 'student_answer')
        pending_submissions = [s for s in pending_submissions if s.status == 'submitted']
        
        results = []
        for submission in pending_submissions:
            try:
                # Call the internal evaluation function
                evaluation_response = evaluate_submission_internal(str(submission._id))
                if evaluation_response.get('success'):
                    results.append({
                        'submission_id': str(submission._id),
                        'student_id': submission.student_id,
                        'roll_number': submission.roll_number,
                        'status': 'evaluated',
                        'score': evaluation_response.get('overall_score', 0)
                    })
                else:
                    results.append({
                        'submission_id': str(submission._id),
                        'student_id': submission.student_id,
                        'roll_number': submission.roll_number,
                        'status': 'error',
                        'error': evaluation_response.get('error', 'Evaluation failed')
                    })
            except Exception as e:
                results.append({
                    'submission_id': str(submission._id),
                    'student_id': submission.student_id,
                    'roll_number': submission.roll_number,
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            "message": f"Batch evaluation completed for {len(pending_submissions)} submissions",
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def evaluate_submission_internal(submission_id):
    """Internal evaluation function for batch processing"""
    try:
        # Get submission
        submission = submission_model.get_by_id(submission_id)
        if not submission:
            return {"success": False, "error": "Submission not found"}
        
        # Only evaluate student answers
        if submission.submission_type != 'student_answer':
            return {"success": False, "error": "Only student answers can be evaluated"}
        
        # Get assignment and answer key
        assignment = assignment_model.get_by_id(submission.assignment_id)
        if not assignment:
            return {"success": False, "error": "Assignment not found"}
        
        # Get teacher's answer key
        answer_key_text = ""
        if assignment.answer_key:
            if submission.roll_number and submission.roll_number in assignment.answer_key:
                answer_key_text = assignment.answer_key[submission.roll_number][0]
            elif 'teacher_key' in assignment.answer_key:
                answer_key_text = assignment.answer_key['teacher_key'][0]
            elif len(assignment.answer_key) > 0:
                first_key = next(iter(assignment.answer_key.values()))
                answer_key_text = first_key[0] if first_key else ""
        
        student_answer = submission.corrected_text[0] if submission.corrected_text else ""
        
        if not student_answer or not answer_key_text:
            return {"success": False, "error": "Missing student answer or answer key"}
        
        # Evaluate using Gemini
        evaluation = gemini_processor.evaluate_answer(
            student_answer=student_answer,
            correct_answer=answer_key_text,
            rubric=assignment.rubric
        )
        
        # Update submission with evaluation results
        submission_model.add_evaluation_results(
            submission_id,
            scores=evaluation,
            feedback=[evaluation.get('feedback', 'No feedback available')],
            total_score=evaluation.get('overall_score', 0)
        )
        
        return {
            "success": True,
            "overall_score": evaluation.get('overall_score', 0),
            "scores": evaluation
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@submissions_bp.route('/api/submissions/student/<student_id>', methods=['GET'])
def get_student_submissions(student_id):
    """Get all submissions by a student"""
    try:
        submissions = submission_model.get_student_submissions(student_id)
        submissions_data = [submission.to_dict() for submission in submissions]
        
        return jsonify({
            "submissions": submissions_data,
            "total": len(submissions_data)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@submissions_bp.route('/api/submissions/teacher/<teacher_id>', methods=['GET'])
def get_teacher_submissions(teacher_id):
    """Get all answer key submissions by a teacher"""
    try:
        submissions = submission_model.get_teacher_submissions(teacher_id)
        submissions_data = [submission.to_dict() for submission in submissions]
        
        return jsonify({
            "submissions": submissions_data,
            "total": len(submissions_data)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@submissions_bp.route('/api/submissions/assignment/<assignment_id>', methods=['GET'])
def get_assignment_submissions(assignment_id):
    """Get all submissions for an assignment (both student answers and teacher keys)"""
    try:
        # Get student submissions
        student_submissions = submission_model.get_by_assignment(assignment_id, 'student_answer')
        # Get teacher answer keys
        teacher_submissions = submission_model.get_by_assignment(assignment_id, 'teacher_answer_key')
        
        all_submissions = student_submissions + teacher_submissions
        submissions_data = [submission.to_dict() for submission in all_submissions]
        
        return jsonify({
            "submissions": submissions_data,
            "total": len(submissions_data),
            "student_answers": len(student_submissions),
            "answer_keys": len(teacher_submissions),
            "evaluated": len([s for s in student_submissions if s.status == 'evaluated']),
            "pending": len([s for s in student_submissions if s.status == 'submitted'])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@submissions_bp.route('/api/submissions/<submission_id>', methods=['GET'])
def get_submission_details(submission_id):
    """Get detailed submission information"""
    try:
        submission = submission_model.get_by_id(submission_id)
        if not submission:
            return jsonify({"error": "Submission not found"}), 404
        
        submission_data = submission.to_dict()
        
        # Get assignment details if available
        if submission.assignment_id:
            assignment = assignment_model.get_by_id(submission.assignment_id)
            if assignment:
                submission_data['assignment_title'] = assignment.title
                submission_data['assignment_subject'] = assignment.subject
                submission_data['upload_mode'] = assignment.upload_mode
        
        # Get user details
        user = user_model.find_by_id(submission.user_id)
        if user:
            submission_data['user_name'] = user.name
            submission_data['user_email'] = user.email
        
        # Get student details if applicable
        if submission.user_role == 'student' and submission.student_id:
            student = student_model.get_by_user_id(submission.student_id)
            if student:
                submission_data['roll_number'] = student.roll_number
                submission_data['class_name'] = student.class_name
        
        return jsonify({
            "submission": submission_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@submissions_bp.route('/api/submissions/<submission_id>/feedback', methods=['POST'])
def add_teacher_feedback(submission_id):
    """Add manual feedback from teacher (only for student answers)"""
    try:
        data = request.json
        feedback = data.get('feedback')
        additional_score = data.get('additional_score')
        
        if not feedback:
            return jsonify({"error": "Feedback is required"}), 400
        
        # Get current submission
        submission = submission_model.get_by_id(submission_id)
        if not submission:
            return jsonify({"error": "Submission not found"}), 404
        
        # Only allow feedback on student answers
        if submission.submission_type != 'student_answer':
            return jsonify({"error": "Can only add feedback to student answers"}), 400
        
        # Update submission with teacher feedback
        update_data = {
            'teacher_feedback': feedback,
            'status': 'evaluated'
        }
        
        if additional_score is not None:
            update_data['total_score'] = submission.total_score + additional_score
        
        submission_model.update_submission(submission_id, update_data)
        
        return jsonify({
            "message": "Feedback added successfully",
            "submission_id": submission_id
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500