from datetime import datetime
from bson import ObjectId
"teacher assingnments and related data"
class Assignment:
    def __init__(self, data):
        
        self._id = data.get('_id', ObjectId())
        self.upload_mode = data.get('upload_mode', 'student')
        self.teacher_id = data['teacher_id']
        self.title = data['title']
        self.subject = data['subject']
        self.questions = data['questions']  # List of question objects
        self.answer_key = data['answer_key']  # Correct answers
        self.total_marks = data.get('total_marks', 0)
        self.due_date = data.get('due_date')
        self.created_at = data.get('created_at', datetime.utcnow())
        self.rubric = data.get('rubric', {
            'meaning_comprehension': 40,
            'key_concepts': 30,
            'technical_accuracy': 20,
            'structure': 10
        })
    
    def to_dict(self):
        return {
            '_id': str(self._id),
            'teacher_id': self.teacher_id,
            'title': self.title,
            'subject': self.subject,
            'questions': self.questions,
            'answer_key': self.answer_key,
            'total_marks': self.total_marks,
            'due_date': self.due_date,
            'created_at': self.created_at,
            'rubric': self.rubric
        }

class AssignmentModel:
    def __init__(self, db):
        self.collection = db.get_collection('assignments')
    
    def create_assignment(self, assignment_data):
        assignment = Assignment(assignment_data)
        result = self.collection.insert_one(assignment.to_dict())
        return str(result.inserted_id)
    
    def get_by_teacher(self, teacher_id):
        assignments = self.collection.find({'teacher_id': teacher_id})
        return [Assignment(assignment) for assignment in assignments]
    
    def get_by_id(self, assignment_id):
        assignment_data = self.collection.find_one({'_id': ObjectId(assignment_id)})
        return Assignment(assignment_data) if assignment_data else None
    
    def update_answer_key(self, assignment_id, roll_number, answers):
        """Update answer key for specific roll number"""
        result = self.collection.update_one(
            {'_id': ObjectId(assignment_id)},
            {'$set': {f'answer_key.{roll_number}': answers}}
        )
        return result.modified_count > 0
    
    def get_answer_key(self, assignment_id, roll_number):
        """Get answer key for specific roll number"""
        assignment_data = self.collection.find_one(
            {'_id': ObjectId(assignment_id)},
            {f'answer_key.{roll_number}': 1}
        )
        return assignment_data.get('answer_key', {}).get(roll_number) if assignment_data else None