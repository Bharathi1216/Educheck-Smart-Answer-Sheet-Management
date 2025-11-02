from flask import Blueprint, request, jsonify
from services.vision_service import VisionService
from services.gemini_service import GeminiService
from services.answer_key_processor import AnswerKeyProcessor
from services.storage_service import StorageService
from models.assignment import AssignmentModel
from models.student import StudentModel
from config.database import db_instance
import os

assignments_bp = Blueprint('assignments', __name__)

# Initialize services
db = db_instance
assignment_model = AssignmentModel(db)
student_model = StudentModel(db)
vision_service = VisionService(os.getenv('GOOGLE_VISION_KEY_PATH'))
gemini_service = GeminiService(os.getenv('GEMINI_API_KEY'))
answer_processor = AnswerKeyProcessor(vision_service, gemini_service)
storage_service = StorageService(os.getenv('STORAGE_TYPE'))

@assignments_bp.route('/api/assignments/create', methods=['POST'])
def create_assignment():
    try:
        data = request.json
        assignment_data = {
            'teacher_id': data['teacher_id'],
            'title': data['title'],
            'subject': data['subject'],
            'questions': data.get('questions', []),
            'students_list': data.get('students_list', []),
            'upload_mode': data.get('upload_mode', 'student'),  # NEW FIELD
            'total_marks': data.get('total_marks', 100),
            'due_date': data.get('due_date')
        }
        
        assignment_id = assignment_model.create_assignment(assignment_data)
        
        return jsonify({
            "message": "Assignment created successfully",
            "assignment_id": assignment_id,
            "upload_mode": assignment_data['upload_mode']  # Return mode
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@assignments_bp.route('/api/assignments/<assignment_id>/upload-answer-key', methods=['POST'])
def upload_answer_key(assignment_id):
    """Upload answer key PDF with roll numbers"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        teacher_id = request.form.get('teacher_id')
        
        # Verify assignment exists and belongs to teacher
        assignment = assignment_model.get_by_id(assignment_id)
        if not assignment or assignment.teacher_id != teacher_id:
            return jsonify({"error": "Assignment not found or unauthorized"}), 404
        
        # Save answer key file
        file_path = storage_service.save_file(file, 'answer_keys')
        
        # Process answer key PDF
        result = answer_processor.process_answer_key_pdf(
            file_path, 
            assignment_id, 
            assignment_model
        )
        
        return jsonify({
            "message": "Answer key processed successfully",
            "assignment_id": assignment_id,
            "processing_result": result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@assignments_bp.route('/api/assignments/teacher/<teacher_id>', methods=['GET'])
def get_teacher_assignments(teacher_id):
    """Get all assignments for a teacher"""
    try:
        assignments = assignment_model.get_by_teacher(teacher_id)
        assignments_data = [assignment.to_dict() for assignment in assignments]
        
        return jsonify({
            "assignments": assignments_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@assignments_bp.route('/api/assignments/<assignment_id>', methods=['GET'])
def get_assignment_details(assignment_id):
    """Get detailed assignment information"""
    try:
        assignment = assignment_model.get_by_id(assignment_id)
        if not assignment:
            return jsonify({"error": "Assignment not found"}), 404
        
        # Get student details for the students list
        students_data = []
        for roll_number in assignment.students_list:
            student = student_model.get_by_roll_number(roll_number)
            if student:
                students_data.append(student.to_dict())
        
        assignment_data = assignment.to_dict()
        assignment_data['students_details'] = students_data
        
        return jsonify({
            "assignment": assignment_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@assignments_bp.route('/api/assignments/<assignment_id>/answer-keys', methods=['GET'])
def get_assignment_answer_keys(assignment_id):
    """Get all answer keys for an assignment"""
    try:
        assignment = assignment_model.get_by_id(assignment_id)
        if not assignment:
            return jsonify({"error": "Assignment not found"}), 404
        
        return jsonify({
            "answer_keys": assignment.answer_key,
            "total_students": len(assignment.answer_key)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
