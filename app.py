from flask import Flask, request, jsonify
from flask_cors import CORS
from config.database import db_instance
from config.settings import settings
from routes.assignments import assignments_bp
from routes.submissions import submissions_bp
from routes.auth import auth_bp
import os
from dotenv import load_dotenv
from services.paper_processor import process_uploads, evaluate_student_answers, generate_feedback_for_students

load_dotenv()

# Resolve Google Vision credentials: require explicit env var; do NOT auto-scan project root for checked-in keys.
gv_direct = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
gv_path_env = os.getenv('GOOGLE_VISION_KEY_PATH') or getattr(settings, 'GOOGLE_VISION_KEY_PATH', None)

resolved = None
# prefer explicit GOOGLE_APPLICATION_CREDENTIALS, then GOOGLE_VISION_KEY_PATH
for candidate in (gv_direct, gv_path_env):
    if candidate:
        candidate_path = candidate if os.path.isabs(candidate) else os.path.abspath(candidate)
        if os.path.exists(candidate_path):
            resolved = candidate_path
            break
        else:
            print(f"Warning: configured credential path does not exist: {candidate_path}")

if resolved:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = resolved
    print(f"Using Google Vision credentials from: {resolved}")
else:
    print("Warning: No valid Google Vision credential path found in environment. "
          "Set GOOGLE_APPLICATION_CREDENTIALS to an absolute path outside the repo.")

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = settings.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = settings.MAX_CONTENT_LENGTH

# Initialize database
db = db_instance.connect()

# Register blueprints
app.register_blueprint(assignments_bp)
app.register_blueprint(submissions_bp)
app.register_blueprint(auth_bp)

@app.route('/api/process-uploads', methods=['POST'])
def api_process_uploads():
    """Trigger processing of files currently in the uploads folder (OCR, parsing, store)."""
    try:
        process_uploads()
        return jsonify({"message": "Processing of uploads completed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate-all', methods=['POST'])
def api_evaluate_all():
    """Trigger evaluation of stored student answers against available answer keys."""
    try:
        evaluate_student_answers()
        return jsonify({"message": "Evaluation completed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-feedback', methods=['POST'])
def api_generate_feedback():
    """Trigger generation of feedback for evaluation results."""
    try:
        generate_feedback_for_students()
        return jsonify({"message": "Feedback generation completed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"message": "EduCheck AI Backend is running!"})

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy", "database": "connected"})

if __name__ == '__main__':
    # Create necessary directories
    directories = ['uploads', 'uploads/assignments', 'uploads/submissions', 'uploads/answer_keys']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    app.run(debug=True, port=5000)