import os
from utils.image_processor import ImageProcessor
from utils.gemini_processor import GeminiProcessor
from models.submission import SubmissionModel
from models.assignment import AssignmentModel
from config.database import db_instance

class EvaluationService:
    def __init__(self):
        self.image_processor = ImageProcessor(
            google_vision_key_path="google-vision-key.json",
            gemini_api_key=os.getenv('GEMINI_API_KEY')
        )
        self.gemini_processor = GeminiProcessor(os.getenv('GEMINI_API_KEY'))
        self.submission_model = SubmissionModel(db_instance)
        self.assignment_model = AssignmentModel(db_instance)
    
    def evaluate_submission(self, submission_id):
        try:
            # Get student submission
            submission = self.submission_model.get_by_id(submission_id)
            if not submission:
                return {"error": "Submission not found"}
            
            # Get assignment and answer key
            assignment = self.assignment_model.get_by_id(submission.assignment_id)
            if not assignment:
                return {"error": "Assignment not found"}
            
            # Get teacher's answer key
            answer_key = assignment.answer_key.get("teacher_key", [""])[0]
            student_answer = submission.corrected_text[0] if submission.corrected_text else ""
            
            # ✅ EVALUATE USING GEMINI
            evaluation = self.gemini_processor.evaluate_answer(
                student_answer=student_answer,
                correct_answer=answer_key,
                rubric=assignment.rubric
            )
            
            # Update submission with evaluation results
            self.submission_model.add_evaluation_results(
                submission_id,
                scores=evaluation,
                feedback=[evaluation.get('feedback', '')],
                total_score=evaluation.get('overall_score', 0)
            )
            
            return {
                "submission_id": submission_id,
                "scores": evaluation,
                "total_score": evaluation.get('overall_score', 0)
            }
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {"error": str(e)}