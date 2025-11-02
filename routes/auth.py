from flask import Blueprint, request, jsonify
from typing import TYPE_CHECKING
import datetime
import os

if TYPE_CHECKING:
	# Editor-only hints to silence Pylance missing-import diagnostics
	import bcrypt  # type: ignore
	import jwt  # type: ignore

# Runtime imports with safe fallbacks
try:
	import bcrypt
except Exception:
	# Minimal fallback to avoid import-time crash in dev environments.
	class _BcryptFallback:
		@staticmethod
		def gensalt():
			raise RuntimeError("bcrypt not installed. Install with `pip install bcrypt` to enable password hashing.")

		@staticmethod
		def hashpw(pw, salt):
			raise RuntimeError("bcrypt not installed. Install with `pip install bcrypt` to enable password hashing.")

		@staticmethod
		def checkpw(pw, hashed):
			raise RuntimeError("bcrypt not installed. Install with `pip install bcrypt` to enable password verification.")

	bcrypt = _BcryptFallback()

try:
	import jwt
except Exception:
	# Minimal fallback for JWT functions to avoid import-time crash.
	class _JWTFallback:
		@staticmethod
		def encode(payload, key, algorithm='HS256'):
			raise RuntimeError("PyJWT not installed. Install with `pip install pyjwt` to enable token creation.")

		@staticmethod
		def decode(token, key, algorithms=None):
			raise RuntimeError("PyJWT not installed. Install with `pip install pyjwt` to enable token verification.")

	jwt = _JWTFallback()

auth_bp = Blueprint('auth', __name__)

# Secret key for tokens (like a special stamp for your passes)
SECRET_KEY = "educheck-secret-key-2023"

# Mock user database (in real app, use MongoDB)
users_db = {}
students_db = {}

@auth_bp.route('/api/auth/register', methods=['POST'])
def register():
    """Student/Teacher gets their school ID card"""
    try:
        data = request.json
        
        # Get user details
        email = data['email']
        password = data['password']
        name = data['name']
        role = data['role']  # 'teacher' or 'student'
        
        # Check if user already has ID card
        if email in users_db:
            return jsonify({"error": "User already registered"}), 400
        
        # STEP 1: Hash password (secure storage)
        # Turn "mypassword123" → "xYz123hashed456..."
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        # STEP 2: Create user account
        user_id = f"user_{len(users_db) + 1}"
        users_db[email] = {
            'user_id': user_id,
            'password': hashed_password.decode('utf-8'),
            'name': name,
            'role': role
        }
        
        # STEP 3: If student, create student profile
        if role == 'student':
            students_db[user_id] = {
                'roll_number': data.get('roll_number', ''),
                'class_name': data.get('class_name', '')
            }
        
        # STEP 4: Generate access token (7-day pass)
        token_payload = {
            'user_id': user_id,
            'role': role,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
        }
        token = jwt.encode(token_payload, SECRET_KEY, algorithm='HS256')
        
        return jsonify({
            "message": "Registration successful!",
            "user_id": user_id,
            "role": role,
            "token": token,  # This is their 7-day access pass
            "name": name
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    """User shows ID card to get daily pass"""
    try:
        data = request.json
        email = data['email']
        password = data['password']
        
        # STEP 1: Check if user exists
        if email not in users_db:
            return jsonify({"error": "Invalid email or password"}), 401
        
        user = users_db[email]
        
        # STEP 2: Verify password
        # Compare: "mypassword123" with stored hash "xYz123hashed456..."
        if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({"error": "Invalid email or password"}), 401
        
        # STEP 3: Generate new 7-day pass
        token_payload = {
            'user_id': user['user_id'],
            'role': user['role'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
        }
        token = jwt.encode(token_payload, SECRET_KEY, algorithm='HS256')
        
        return jsonify({
            "message": "Login successful!",
            "token": token,  # New 7-day pass
            "user": {
                'user_id': user['user_id'],
                'name': user['name'],
                'role': user['role']
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/api/auth/verify', methods=['GET'])
def verify_token():
    """Check if pass is still valid"""
    try:
        # Get token from header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "No token provided"}), 401
        
        token = auth_header.replace('Bearer ', '')
        
        # STEP 1: Decode and verify token
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired - please login again"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        
        # STEP 2: Token is valid - return user info
        user_id = payload['user_id']
        user = None
        for u in users_db.values():
            if u['user_id'] == user_id:
                user = u
                break
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        return jsonify({
            "valid": True,
            "user": {
                'user_id': user['user_id'],
                'name': user['name'],
                'role': user['role']
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500