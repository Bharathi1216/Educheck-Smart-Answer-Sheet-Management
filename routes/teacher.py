from flask import Blueprint, request, jsonify
from models.assignment import AssignmentModel
from models.submission import SubmissionModel
from models.student import StudentModel
from config.database import db_instance

teacher_bp = Blueprint('teacher', __name__)

db = db_instance
assignment_model = AssignmentModel(db)
submission_model = SubmissionModel(db)
student_model = StudentModel(db)

@teacher_bp.route('/api/teacher/<teacher_id>/dashboard', methods=['GET'])
def teacher_dashboard(teacher_id):
    """Get teacher dashboard data"""
    try:
        # Get teacher's assignments
        assignments = assignment_model.get_by_teacher(teacher_id)
        
        dashboard_data = {
            "total_assignments": len(assignments),
            "recent_submissions": 0,
            "pending_evaluations": 0,
            "recent_activity": []
        }
        
        # Calculate statistics
        for assignment in assignments:
            # Get submissions for this assignment
            submissions = submission_model.get_by_assignment(str(assignment._id))
            dashboard_data["recent_submissions"] += len(submissions)
            
            # Count pending evaluations
            pending = [s for s in submissions if s.status == 'submitted']
            dashboard_data["pending_evaluations"] += len(pending)
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@teacher_bp.route('/api/teacher/<teacher_id>/assignments-with-submissions', methods=['GET'])
def get_assignments_with_submissions(teacher_id):
    """Get assignments with submission counts"""
    try:
        assignments = assignment_model.get_by_teacher(teacher_id)
        result = []
        
        for assignment in assignments:
            submissions = submission_model.get_by_assignment(str(assignment._id))
            evaluated = [s for s in submissions if s.status == 'evaluated']
            
            assignment_data = assignment.to_dict()
            assignment_data.update({
                "total_submissions": len(submissions),
                "evaluated_submissions": len(evaluated),
                "pending_evaluations": len(submissions) - len(evaluated)
            })
            result.append(assignment_data)
        
        return jsonify({"assignments": result})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    