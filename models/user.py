from datetime import datetime
from bson import ObjectId
"Manages user authentication and profile data"
class User:
    def __init__(self, data):
        self._id = data.get('_id', ObjectId())
        self.email = data['email']
        self.name = data['name']
        self.role = data['role']  # 'teacher' or 'student'
        self.created_at = data.get('created_at', datetime.utcnow())
        self.updated_at = data.get('updated_at', datetime.utcnow())
    
    def to_dict(self):
        return {
            '_id': str(self._id),
            'email': self.email,
            'name': self.name,
            'role': self.role,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

class UserModel:
    def __init__(self, db):
        self.collection = db.get_collection('users')
    
    def create_user(self, user_data):
        user = User(user_data)
        result = self.collection.insert_one(user.to_dict())
        return str(result.inserted_id)
    
    def find_by_email(self, email):
        user_data = self.collection.find_one({'email': email})
        return User(user_data) if user_data else None
    
    def find_by_id(self, user_id):
        user_data = self.collection.find_one({'_id': ObjectId(user_id)})
        return User(user_data) if user_data else None