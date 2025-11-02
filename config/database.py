from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        
    def connect(self):
        """Connect to MongoDB"""
        try:
            # For local MongoDB: mongodb://localhost:27017
            # For MongoDB Atlas: use connection string from .env
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/educheck')
            self.client = MongoClient(mongo_uri)
            self.db = self.client[os.getenv('DB_NAME', 'educheck')]
            print("✅ Connected to MongoDB successfully")
            return self.db
        except Exception as e:
            print(f"❌ MongoDB connection error: {e}")
            raise e
    
    def get_collection(self, collection_name):
        """Get specific collection"""
        if self.db is None:
            self.connect()
        return self.db[collection_name]
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()

# Global database instance
db_instance = Database()