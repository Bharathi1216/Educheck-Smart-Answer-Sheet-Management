import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/educheck')
    DB_NAME = os.getenv('DB_NAME', 'educheck')
    
    # Google APIs
    GOOGLE_VISION_KEY_PATH = os.getenv('GOOGLE_VISION_KEY_PATH', 'google-vision-key.json')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Storage
    STORAGE_TYPE = os.getenv('STORAGE_TYPE', 'local')  # 'local', 'firebase', 's3'
    FIREBASE_CONFIG = os.getenv('FIREBASE_CONFIG', '{}')
    
    # Application
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

settings = Settings()