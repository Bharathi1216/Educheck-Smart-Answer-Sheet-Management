from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	# Editor-only hint for bson
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
          
class Student:
    def __init__(self, data):
        self._id = data.get('_id', ObjectId())
        self.user_id = data['user_id']  # Reference to user collection
        self.roll_number = data['roll_number']
        self.class_name = data.get('class_name', '')
        self.section = data.get('section', '')
        self.enrolled_subjects = data.get('enrolled_subjects', [])
        self.created_at = data.get('created_at', datetime.utcnow())
    
    def to_dict(self):
        return {
            '_id': str(self._id),
            'user_id': self.user_id,
            'roll_number': self.roll_number,
            'class_name': self.class_name,
            'section': self.section,
            'enrolled_subjects': self.enrolled_subjects,
            'created_at': self.created_at
        }

class StudentModel:
    def __init__(self, db):
        self.collection = db.get_collection('students')
    
    def create_student(self, student_data):
        student = Student(student_data)
        result = self.collection.insert_one(student.to_dict())
        return str(result.inserted_id)
    
    def get_by_roll_number(self, roll_number):
        student_data = self.collection.find_one({'roll_number': roll_number})
        return Student(student_data) if student_data else None
    
    def get_by_user_id(self, user_id):
        student_data = self.collection.find_one({'user_id': user_id})
        return Student(student_data) if student_data else None
    
    def get_students_by_classroom(self, class_name, section=None):
        """Get all students in a classroom"""
        query = {'class_name': class_name}
        if section:
            query['section'] = section
            
        students = self.collection.find(query)
        return [Student(student) for student in students]
    
    def update_student(self, user_id, update_data):
        """Update student information"""
        result = self.collection.update_one(
            {'user_id': user_id},
            {'$set': update_data}
        )
        return result.modified_count > 0
    
    def delete_student(self, user_id):
        """Delete student profile"""
        result = self.collection.delete_one({'user_id': user_id})
        return result.deleted_count > 0
    
    def get_all_students(self, limit=100):
        """Get all students with pagination"""
        students = self.collection.find().limit(limit)
        return [Student(student) for student in students]
    