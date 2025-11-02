from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from bson import ObjectId  # type: ignore

try:
	from bson import ObjectId
except Exception:
	import uuid
	class ObjectId(str):
		def __new__(cls, val=None):
			if val is None:
				return str.__new__(cls, uuid.uuid4().hex)
			return str.__new__(cls, str(val))

class Submission:
    def __init__(self, data):
        """
        Represents a student answer submission with comprehensive tracking
        Now supports both student submissions and teacher answer key uploads
        """
        self.uploaded_by_teacher = data.get('uploaded_by_teacher', False)
        self.teacher_uploader_id = data.get('teacher_uploader_id')
        
        self._id = data.get('_id', ObjectId())
        self.submission_type = data['submission_type']  # 'student_answer' or 'teacher_answer_key'
        self.user_id = data['user_id']  # ID of user who submitted (student or teacher)
        self.user_role = data['user_role']  # 'student' or 'teacher'
        
        # For student submissions
        self.student_id = data.get('student_id')  # Reference to student profile
        self.roll_number = data.get('roll_number')  # Direct roll number for quick access
        
        # For teacher answer key submissions
        self.teacher_id = data.get('teacher_id')  # Reference to teacher profile
        self.assignment_id = data.get('assignment_id')  # Linked assignment
        
        # Content storage
        self.answers = data.get('answers', [])  # Processed answers array
        self.extracted_text = data.get('extracted_text', [])  # Raw OCR output
        self.corrected_text = data.get('corrected_text', [])  # Gemini-corrected text
        self.original_filename = data.get('original_filename')  # Original PDF/image name
        self.file_path = data.get('file_path')  # Path to stored file
        
        # Evaluation results (for student submissions)
        self.scores = data.get('scores', {})  # Individual dimension scores
        self.feedback = data.get('feedback', [])  # Detailed feedback per question
        self.total_score = data.get('total_score', 0)  # Overall score
        self.grade = data.get('grade')  # Letter grade if applicable
        
        # Status tracking
        self.status = data.get('status', 'submitted')  # submitted, processing, evaluated, error
        self.processing_stage = data.get('processing_stage', 'uploaded')  # Track pipeline progress
        
        # Timestamps for audit trail
        self.submitted_at = data.get('submitted_at', datetime.utcnow())
        self.processed_at = data.get('processed_at')  # When OCR completed
        self.corrected_at = data.get('corrected_at')  # When Gemini processing completed
        self.evaluated_at = data.get('evaluated_at')  # When evaluation completed
        
        # Metadata
        self.page_count = data.get('page_count', 1)  # Number of pages in PDF
        self.file_size = data.get('file_size')  # File size in bytes
        self.upload_source = data.get('upload_source', 'web')  # web, mobile, api

    def to_dict(self):
        """Convert submission object to dictionary for database storage"""
        return {
            '_id': str(self._id),
            'submission_type': self.submission_type,
            'user_id': self.user_id,
            'user_role': self.user_role,
            'student_id': self.student_id,
            'roll_number': self.roll_number,
            'teacher_id': self.teacher_id,
            'assignment_id': self.assignment_id,
            'answers': self.answers,
            'extracted_text': self.extracted_text,
            'corrected_text': self.corrected_text,
            'original_filename': self.original_filename,
            'file_path': self.file_path,
            'scores': self.scores,
            'feedback': self.feedback,
            'total_score': self.total_score,
            'grade': self.grade,
            'status': self.status,
            'processing_stage': self.processing_stage,
            'submitted_at': self.submitted_at,
            'processed_at': self.processed_at,
            'corrected_at': self.corrected_at,
            'evaluated_at': self.evaluated_at,
            'page_count': self.page_count,
            'file_size': self.file_size,
            'upload_source': self.upload_source
        }
    
    def update_processing_stage(self, stage):
        """Update the processing stage and corresponding timestamp"""
        self.processing_stage = stage
        if stage == 'text_extracted':
            self.processed_at = datetime.utcnow()
        elif stage == 'text_corrected':
            self.corrected_at = datetime.utcnow()
        elif stage == 'evaluated':
            self.evaluated_at = datetime.utcnow()
            self.status = 'evaluated'


class SubmissionModel:
    def __init__(self, db):
        """Initialize submission model with database connection"""
        self.collection = db.get_collection('submissions')
    
    def create_submission(self, submission_data):
        """
        Create a new submission record
        Used for both student answers and teacher answer keys
        """
        submission = Submission(submission_data)
        result = self.collection.insert_one(submission.to_dict())
        return str(result.inserted_id)
    
    def get_by_id(self, submission_id):
        """Get submission by its unique ID"""
        submission_data = self.collection.find_one({'_id': ObjectId(submission_id)})
        return Submission(submission_data) if submission_data else None
    
    def get_by_student_assignment(self, student_id, assignment_id):
        """
        Get student's submission for a specific assignment
        Used to prevent duplicate submissions
        """
        submission_data = self.collection.find_one({
            'student_id': student_id,
            'assignment_id': assignment_id,
            'submission_type': 'student_answer'
        })
        return Submission(submission_data) if submission_data else None
    
    def get_by_teacher_assignment(self, teacher_id, assignment_id, submission_type='teacher_answer_key'):
        """
        Get teacher's answer key submission for a specific assignment
        Teachers can upload multiple answer keys (updated versions)
        """
        submissions = self.collection.find({
            'teacher_id': teacher_id,
            'assignment_id': assignment_id,
            'submission_type': submission_type
        }).sort('submitted_at', -1)  # Get most recent first
        
        return [Submission(submission) for submission in submissions]
    
    def get_by_roll_number(self, roll_number, assignment_id=None):
        """
        Get submissions by roll number
        Useful for finding all submissions from a particular student
        """
        query = {'roll_number': roll_number, 'submission_type': 'student_answer'}
        if assignment_id:
            query['assignment_id'] = assignment_id
            
        submissions = self.collection.find(query)
        return [Submission(submission) for submission in submissions]
    
    def get_by_assignment(self, assignment_id, submission_type=None):
        """
        Get all submissions for an assignment
        Teacher can view all student submissions for their assignment
        """
        query = {'assignment_id': assignment_id}
        if submission_type:
            query['submission_type'] = submission_type
            
        submissions = self.collection.find(query)
        return [Submission(submission) for submission in submissions]
    
    def get_student_submissions(self, student_id, limit=10):
        """
        Get recent submissions by a student across all assignments
        For student dashboard and progress tracking
        """
        submissions = self.collection.find({
            'student_id': student_id,
            'submission_type': 'student_answer'
        }).sort('submitted_at', -1).limit(limit)
        
        return [Submission(submission) for submission in submissions]
    
    def get_teacher_submissions(self, teacher_id, submission_type='teacher_answer_key', limit=10):
        """
        Get recent answer key uploads by a teacher
        For teacher's upload history and management
        """
        submissions = self.collection.find({
            'teacher_id': teacher_id,
            'submission_type': submission_type
        }).sort('submitted_at', -1).limit(limit)
        
        return [Submission(submission) for submission in submissions]
    
    def update_submission(self, submission_id, update_data):
        """
        Update submission data - used throughout processing pipeline
        Supports partial updates for different processing stages
        """
        # Add timestamp for the update
        update_data['updated_at'] = datetime.utcnow()
        
        result = self.collection.update_one(
            {'_id': ObjectId(submission_id)},
            {'$set': update_data}
        )
        return result.modified_count > 0
    
    def update_processing_stage(self, submission_id, stage):
        """Convenience method to update processing stage with proper timestamps"""
        submission = self.get_by_id(submission_id)
        if submission:
            submission.update_processing_stage(stage)
            return self.update_submission(submission_id, {
                'processing_stage': submission.processing_stage,
                'processed_at': submission.processed_at,
                'corrected_at': submission.corrected_at,
                'evaluated_at': submission.evaluated_at,
                'status': submission.status
            })
        return False
    
    def add_evaluation_results(self, submission_id, scores, feedback, total_score, grade=None):
        """
        Store evaluation results for a student submission
        Called after AI evaluation is complete
        """
        update_data = {
            'scores': scores,
            'feedback': feedback,
            'total_score': total_score,
            'status': 'evaluated',
            'evaluated_at': datetime.utcnow(),
            'processing_stage': 'evaluated'
        }
        
        if grade:
            update_data['grade'] = grade
            
        return self.update_submission(submission_id, update_data)
    
    def get_pending_evaluations(self, assignment_id=None, limit=50):
        """
        Get submissions that are ready for evaluation but not yet processed
        For batch processing and background jobs
        """
        query = {
            'status': 'submitted',
            'submission_type': 'student_answer',
            'processing_stage': 'text_corrected'  # Ready for evaluation
        }
        
        if assignment_id:
            query['assignment_id'] = assignment_id
            
        submissions = self.collection.find(query).limit(limit)
        return [Submission(submission) for submission in submissions]
    
    def get_processing_stats(self, assignment_id=None):
        """
        Get statistics about submission processing status
        For admin dashboard and monitoring
        """
        pipeline = [
            {'$match': {'submission_type': 'student_answer'}}
        ]
        
        if assignment_id:
            pipeline[0]['$match']['assignment_id'] = assignment_id
            
        pipeline.extend([
            {
                '$group': {
                    '_id': '$status',
                    'count': {'$sum': 1},
                    'avg_score': {'$avg': '$total_score'}
                }
            }
        ])
        
        return list(self.collection.aggregate(pipeline))
    
    def delete_submission(self, submission_id):
        """
        Delete a submission (with caution)
        For admin cleanup or user request
        """
        result = self.collection.delete_one({'_id': ObjectId(submission_id)})
        return result.deleted_count > 0