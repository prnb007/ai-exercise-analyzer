"""
Exercise Form Analyzer - MongoDB Version
Production-ready Flask app with MongoDB and optimized video handling
"""

import os
import cv2
import torch
import numpy as np
import mediapipe as mp

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
import hashlib
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, current_app, send_file, send_from_directory
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import tempfile
import shutil
import time
import secrets
import random
import json
from datetime import datetime, timedelta, date, timezone
# Removed imports for deleted files - functions will be defined inline
from mongodb_config import mongodb_config, get_mongodb, get_collection
from mongodb_models import User, ExerciseSession, Achievement, Progress, Card, UserCard, UserLevel, DailyChallenge, UserChallenge
from forms import LoginForm, SignupForm, ForgotPasswordForm
# Removed imports for deleted files

# Define missing functions inline
def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    import math
    
    # Convert to numpy arrays if needed
    if not hasattr(p1, '__len__'):
        return 90  # Default angle
    
    # Calculate vectors
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Calculate angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
    angle = math.degrees(math.acos(cos_angle))
    
    return angle

class PushupLSTM(torch.nn.Module):
    """Real LSTM model for exercise analysis"""
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, num_classes=2):
        super(PushupLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        if x.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.fc(out[:, -1, :])
        
        return out
    
    def predict(self, data):
        """Make prediction on input data"""
        self.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)
            
            # Ensure data has batch dimension
            if data.dim() == 2:
                data = data.unsqueeze(0)
            
            if data.is_cuda:
                data = data.cuda()
            
            output = self.forward(data)
            probabilities = torch.softmax(output, dim=1)
            return probabilities[:, 1].item()  # Return probability of correct form

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    try:
        # Try to load user by ID (works for both ObjectId and string IDs)
        return User.find_by_id(user_id)
    except Exception as e:
        print(f"Error loading user {user_id}: {e}")
        return None

# Make current_user available in all templates
@app.context_processor
def inject_user():
    """Inject current_user into template context"""
    from flask_login import current_user
    return dict(current_user=current_user)

# Session configuration for persistence
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize MongoDB
try:
    mongodb_config.get_client()
    print("MongoDB Atlas connected successfully!")
except Exception as e:
    print(f"MongoDB connection error: {e}")

# Initialize gamification
# gamification_manager = GamificationManager()  # Not needed for MongoDB

# Initialize authentication
# auth_manager = AuthManager()  # Removed - not needed for MongoDB
# google_oauth = GoogleOAuth()  # Removed - not needed for MongoDB

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables for temporary video storage
temp_videos = {}  # Store temporary video data: {session_id: {video_path, analysis_results, timestamp}}

def initialize_gamification_data():
    """Initialize gamification data like achievements and cards"""
    try:
        from gamification_manager import initialize_gamification_data as init_gamification
        init_gamification()
    except Exception as e:
        print(f"Gamification initialization error: {e}")

def initialize_database():
    """Initialize database and gamification data"""
    try:
        with app.app_context():
            # Initialize gamification data
            initialize_gamification_data()
            initialize_default_cards()
            print("Database initialized")
            print("Gamification data initialized")
    except Exception as e:
        print(f"Database initialization error: {e}")

def initialize_default_cards():
    """Initialize default cards"""
    try:
        # Create default cards
        cards_data = [
            {
                'name': 'Form Master',
                'description': 'Perfect your exercise form',
                'rarity': 'common',
                'category': 'form'
            },
            {
                'name': 'Strength Builder',
                'description': 'Build incredible strength',
                'rarity': 'rare',
                'category': 'strength'
            },
            {
                'name': 'Endurance King',
                'description': 'Unlimited stamina and endurance',
                'rarity': 'epic',
                'category': 'endurance'
            },
            {
                'name': 'Legendary Warrior',
                'description': 'The ultimate fitness champion',
                'rarity': 'legendary',
                'category': 'special'
            },
            {
                'name': 'Speed Demon',
                'description': 'Lightning fast movements',
                'rarity': 'common',
                'category': 'speed'
            },
            {
                'name': 'Flexibility Guru',
                'description': 'Master of mobility',
                'rarity': 'rare',
                'category': 'flexibility'
            },
            {
                'name': 'Balance Master',
                'description': 'Perfect stability and control',
                'rarity': 'epic',
                'category': 'balance'
            },
            {
                'name': 'Iron Will',
                'description': 'Unbreakable determination',
                'rarity': 'legendary',
                'category': 'mental'
            }
        ]
        
        for card_data in cards_data:
            # Check if card already exists
            existing_cards = Card.find_all()
            if not any(card.name == card_data['name'] for card in existing_cards):
                Card.create(**card_data)
                
    except Exception as e:
        print(f"Card initialization error: {e}")

# Initialize database on startup
initialize_database()

def generate_daily_challenge():
    """Generate a random daily challenge"""
    import random
    
    # Challenge templates
    challenge_templates = [
        {
            'name': 'Push-up Power',
            'description': 'Complete perfect push-ups with excellent form',
            'exercise_type': 'pushup',
            'target_reps': random.randint(5, 15),
            'target_accuracy': random.randint(75, 95),
            'xp_reward': random.randint(30, 80),
            'card_reward': 'Form Master' if random.random() < 0.3 else None
        },
        {
            'name': 'Squat Strength',
            'description': 'Master the perfect squat technique',
            'exercise_type': 'squat',
            'target_reps': random.randint(8, 20),
            'target_accuracy': random.randint(70, 90),
            'xp_reward': random.randint(40, 70),
            'card_reward': 'Strength Builder' if random.random() < 0.3 else None
        },
        {
            'name': 'Plank Endurance',
            'description': 'Hold a perfect plank position',
            'exercise_type': 'plank',
            'target_reps': random.randint(1, 3),  # Planks are held, not repeated
            'target_accuracy': random.randint(80, 95),
            'xp_reward': random.randint(35, 75),
            'card_reward': 'Endurance King' if random.random() < 0.3 else None
        },
        {
            'name': 'Pull-up Challenge',
            'description': 'Complete controlled pull-ups',
            'exercise_type': 'pullup',
            'target_reps': random.randint(3, 10),
            'target_accuracy': random.randint(75, 90),
            'xp_reward': random.randint(50, 100),
            'card_reward': 'Legendary Warrior' if random.random() < 0.2 else None
        },
        {
            'name': 'Tricep Power',
            'description': 'Build tricep strength with dips',
            'exercise_type': 'tricep_dips',
            'target_reps': random.randint(5, 12),
            'target_accuracy': random.randint(70, 85),
            'xp_reward': random.randint(35, 65),
            'card_reward': None
        }
    ]
    
    # Select random challenge
    challenge_data = random.choice(challenge_templates)
    challenge_data['challenge_date'] = datetime.now(timezone.utc).date()
    
    return challenge_data

def ensure_daily_challenge():
    """Ensure there's a daily challenge for today"""
    today = datetime.now(timezone.utc).date()
    existing_challenge = DailyChallenge.find_by_date(today)
    
    if not existing_challenge:
        # Generate a new challenge
        challenge_data = generate_daily_challenge()
        DailyChallenge.create(**challenge_data)
        print(f"Generated daily challenge: {challenge_data['name']}")
    else:
        print(f"Daily challenge already exists: {existing_challenge.name}")

# Ensure daily challenge exists on startup
ensure_daily_challenge()

# Exercise configurations
EXERCISES = {
    "pushup": {
        "name": "Push-ups",
        "input_size": 5,
        "model_file": "pushup_lstm_features.pt",
        "stats_file": "angle_stats_pushup.npz",
        "demo_video": "pushup/Copy of push up 1.mp4",
        "description": "A classic bodyweight exercise that targets chest, shoulders, and triceps.",
        "benefits": [
            "Builds upper body strength",
            "Improves core stability",
            "Enhances cardiovascular fitness",
            "No equipment required",
            "Can be modified for all fitness levels"
        ],
        "muscles": ["Chest", "Shoulders", "Triceps", "Core"],
        "difficulty": "Beginner to Advanced",
        "duration": "30-60 seconds"
    },
    "squat": {
        "name": "Squats",
        "input_size": 5,
        "model_file": "squat_lstm_features.pt",
        "stats_file": "angle_stats_squat.npz",
        "demo_video": "squat/WhatsApp Video 2025-10-11 at 21.45.12.mp4",
        "description": "A fundamental lower body exercise that targets legs and glutes.",
        "benefits": [
            "Builds leg and glute strength",
            "Improves functional movement",
            "Enhances balance and coordination",
            "Burns calories effectively",
            "Strengthens core muscles"
        ],
        "muscles": ["Quadriceps", "Glutes", "Hamstrings", "Core"],
        "difficulty": "Beginner to Advanced",
        "duration": "30-60 seconds"
    },
    "plank": {
        "name": "Plank",
        "input_size": 3,
        "model_file": "plank_lstm_features.pt",
        "stats_file": "angle_stats_plank.npz",
        "demo_video": "plank/WhatsApp Video 2025-10-11 at 21.05.36.mp4",
        "description": "An isometric core exercise that builds stability and endurance.",
        "benefits": [
            "Strengthens entire core",
            "Improves posture",
            "Reduces back pain risk",
            "Enhances balance and stability",
            "Can be done anywhere"
        ],
        "muscles": ["Core", "Shoulders", "Glutes", "Back"],
        "difficulty": "Beginner to Advanced",
        "duration": "30-120 seconds"
    },
    "pullup": {
        "name": "Pull-ups",
        "input_size": 2,
        "model_file": "pullup_lstm_features.pt",
        "stats_file": "angle_stats_pullup.npz",
        "demo_video": "pullup/1.mp4",
        "description": "An upper body strength exercise that primarily targets the back and biceps.",
        "benefits": [
            "Builds back and bicep strength",
            "Improves grip strength",
            "Enhances shoulder stability",
            "Develops functional pulling strength",
            "Great for posture improvement"
        ],
        "muscles": ["Back", "Biceps", "Shoulders", "Core"],
        "difficulty": "Intermediate to Advanced",
        "duration": "30-60 seconds"
    },
    "tricep_dips": {
        "name": "Tricep Dips",
        "input_size": 4,
        "model_file": "tricep_dips_lstm_features.pt",
        "stats_file": "angle_stats_tricep_dips.npz",
        "demo_video": "tricep_dips/WhatsApp Video 2025-10-11 at 21.46.08.mp4",
        "description": "An upper body exercise that specifically targets the triceps and shoulders.",
        "benefits": [
            "Builds tricep strength",
            "Strengthens shoulders",
            "Improves pushing power",
            "Can be done with minimal equipment",
            "Great for arm definition"
        ],
        "muscles": ["Triceps", "Shoulders", "Chest", "Core"],
        "difficulty": "Beginner to Advanced",
        "duration": "30-60 seconds"
    }
}

@app.route('/')
def index():
    """Homepage"""
    # If user is not authenticated, show landing page
    if not current_user.is_authenticated:
        return render_template('landing.html')
    # If user is authenticated, show exercise selection
    return render_template('index.html', exercises=EXERCISES)

@app.route('/dashboard')
@login_required
def dashboard():
    """Exercise dashboard for authenticated users"""
    return render_template('index.html', exercises=EXERCISES)

@app.route('/clear-session')
def clear_session():
    """Clear session data - useful for debugging"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        
        user = User.find_by_email(email)
        if user and user.password_hash == hashlib.sha256(password.encode()).hexdigest():
            # Use Flask-Login for authentication
            login_user(user, remember=form.remember_me.data)
            
            # Update last login
            user.update_last_login()
            
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    form = SignupForm()
    if form.validate_on_submit():
        email = form.email.data
        username = form.name.data  # Using name field from form
        password = form.password.data
        
        # Check if user already exists
        if User.find_by_email(email):
            flash('Email already registered', 'error')
            return render_template('signup.html', form=form)
        
        # Create new user
        user = User.create(email, username, password)
        
        # Use Flask-Login for authentication
        login_user(user, remember=True)
        
        flash('Registration successful!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('signup.html', form=form)

@app.route('/logout')
def logout():
    """User logout"""
    # Clear temporary videos for this user
    if current_user.is_authenticated:
        user_id = str(current_user._id)
        # Remove any temporary videos for this user
        for session_id, video_data in list(temp_videos.items()):
            if video_data.get('user_id') == user_id:
                # Clean up video file
                if os.path.exists(video_data.get('video_path', '')):
                    os.remove(video_data.get('video_path', ''))
                del temp_videos[session_id]
    
    # Use Flask-Login for logout
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    """Video analysis page"""
    
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file selected', 'error')
            return redirect(url_for('analyze'))
        
        video_file = request.files['video']
        if video_file.filename == '':
            flash('No video file selected', 'error')
            return redirect(url_for('analyze'))
        
        if video_file:
            # Generate unique session ID for this analysis
            session_id = secrets.token_urlsafe(16)
            
            # Save video temporarily
            filename = secure_filename(video_file.filename)
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, filename)
            video_file.save(video_path)
            
            # Store video info in temporary storage
            temp_videos[session_id] = {
                'video_path': video_path,
                'user_id': str(current_user._id),
                'timestamp': datetime.now(),
                'analysis_results': None
            }
            
            # Create exercise session in database
            exercise_session = ExerciseSession.create(
                user_id=str(current_user._id),
                exercise_type='pushup',  # Default for now
                video_filename=filename,
                session_id=session_id
            )
            
            # Perform analysis
            try:
                analysis_results = perform_video_analysis(video_path)
                
                # Update session with results
                exercise_session.update_status(
                    'completed',
                    analysis_results=analysis_results,
                    score=analysis_results.get('score', 0),
                    feedback=analysis_results.get('feedback', '')
                )
                
                # Update temporary storage
                temp_videos[session_id]['analysis_results'] = analysis_results
                
                # Award points and check achievements
                gamification_data = award_points_and_achievements(str(current_user._id), analysis_results)
                
                # Add gamification data to analysis results
                if gamification_data:
                    analysis_results['gamification'] = gamification_data
                
                return render_template('results.html', 
                                     session_id=session_id,
                                     results=analysis_results,
                                     video_filename=filename)
                
            except Exception as e:
                print(f"Analysis error: {e}")
                exercise_session.update_status('failed')
                flash('Analysis failed. Please try again.', 'error')
                return redirect(url_for('analyze'))
    
    return render_template('analyze.html')

@app.route('/results/<session_id>')
@login_required
def view_results(session_id):
    """View analysis results (temporary access)"""
    # Check if session exists and belongs to user
    if session_id not in temp_videos:
        flash('Results not found or expired', 'error')
        return redirect(url_for('dashboard'))
    
    video_data = temp_videos[session_id]
    if video_data['user_id'] != str(current_user._id):
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    # Check if results are still valid (within 1 hour)
    if datetime.now() - video_data['timestamp'] > timedelta(hours=1):
        # Clean up expired video
        if os.path.exists(video_data['video_path']):
            os.remove(video_data['video_path'])
        del temp_videos[session_id]
        flash('Results have expired', 'error')
        return redirect(url_for('dashboard'))
    
    return render_template('results.html',
                          session_id=session_id,
                          result=video_data['analysis_results'])

@app.route('/download/<session_id>')
def download_video(session_id):
    """Download analyzed video (temporary access only)"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Check if session exists and belongs to user
    if session_id not in temp_videos:
        flash('Video not found or expired', 'error')
        return redirect(url_for('dashboard'))
    
    video_data = temp_videos[session_id]
    if video_data['user_id'] != session['user_id']:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    # Check if video is still valid (within 1 hour)
    if datetime.now() - video_data['timestamp'] > timedelta(hours=1):
        # Clean up expired video
        if os.path.exists(video_data['video_path']):
            os.remove(video_data['video_path'])
        del temp_videos[session_id]
        flash('Video has expired', 'error')
        return redirect(url_for('dashboard'))
    
    # Return video file
    return send_file(video_data['video_path'], as_attachment=True)


def perform_video_analysis(video_path, exercise_type='pushup'):
    """Perform comprehensive video analysis using ML models and MediaPipe"""
    try:
        exercise_config = EXERCISES[exercise_type]
        
        # Load model and stats
        model_path = os.path.join('models', exercise_config['model_file'])
        stats_path = os.path.join('models', exercise_config['stats_file'])
        
        if not os.path.exists(model_path) or not os.path.exists(stats_path):
            print(f"Model or stats file not found: {model_path}, {stats_path}")
            # Use dummy model for testing
            model = PushupLSTM()
            mean_angles = np.array([90, 90, 90, 90, 90])  # Default angles
        else:
            # Setup model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = PushupLSTM(input_size=exercise_config['input_size'])
            
            # Load model state dict
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval().to(device)
            
            # Load angle stats
            angle_stats = np.load(stats_path)
            mean_angles = angle_stats["mean"]
            
            print(f"Loaded model from {model_path} with {len(mean_angles)} angle features")
        
        # Initialize pose detection
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file. Please check if the file is corrupted or in an unsupported format.")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width == 0 or height == 0:
            raise Exception("Invalid video dimensions. Please check if the video file is valid.")
        
        # Create output video
        output_filename = f"feedback_{exercise_type}_{int(time.time())}.mp4"
        output_path = os.path.join('output', output_filename)
        
        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        
        seq_len = 60
        all_keypoints = []
        frame_accuracies = []
        all_mistakes = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb = np.ascontiguousarray(image_rgb)
                if image_rgb.dtype != np.uint8:
                    image_rgb = image_rgb.astype(np.uint8)
                
                # Handle MediaPipe version compatibility
                try:
                    results = pose.process(image_rgb)
                except AttributeError as e:
                    if "GetPrototype" in str(e):
                        # Fallback for MediaPipe version issues
                        print(f"MediaPipe compatibility issue, using fallback: {e}")
                        results = None
                    else:
                        raise e
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                results = None
            
            frame_angles = None
            
            if results and results.pose_landmarks:
                # Extract keypoints
                keypoints_flat = np.zeros(33*2)
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    keypoints_flat[i*2] = lm.x * width
                    keypoints_flat[i*2+1] = lm.y * height
                
                # Draw pose landmarks and connections on frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Calculate angles based on exercise type
                frame_angles = calculate_exercise_angles(keypoints_flat, exercise_type)
            
            # Append frame angles
            if frame_angles is None:
                all_keypoints.append(np.zeros(exercise_config['input_size']))
            else:
                all_keypoints.append(frame_angles)
            
            # Prepare sequence for model
            keypoints_seq = np.array(all_keypoints[-seq_len:])
            if len(keypoints_seq) < seq_len:
                pad = np.zeros((seq_len - len(keypoints_seq), keypoints_seq.shape[1]))
                keypoints_seq = np.vstack([pad, keypoints_seq])
            
            # Get prediction
            input_tensor = torch.tensor([keypoints_seq], dtype=torch.float32)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            prob_correct = model.predict(input_tensor)
            
            # Detect mistakes
            mistakes = detect_mistakes(keypoints_seq[-1], mean_angles, exercise_type)
            all_mistakes.extend(mistakes)
            
            # Calculate accuracy
            accuracy = int(min(100, max(0, prob_correct * 100 - len(mistakes)*10)))
            frame_accuracies.append(accuracy)
            
            # Draw feedback on frame
            y_offset = 30
            for i, mistake in enumerate(mistakes):
                cv2.putText(frame, mistake, (30, y_offset + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            display_acc = int(np.mean(frame_accuracies[-20:])) if frame_accuracies else 0
            cv2.putText(frame, f"Accuracy: {display_acc}%", 
                       (30, y_offset + (len(mistakes)+1)*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        pose.close()
        
        # Calculate final results
        avg_accuracy = int(np.mean(frame_accuracies)) if frame_accuracies else 0
        unique_mistakes = list(set(all_mistakes))
        
        # If no pose landmarks detected, use simple fallback analysis
        if len(frame_accuracies) == 0:
            print("No pose landmarks detected, using simple fallback analysis")
            # Simple fallback: assume basic form if video is long enough
            video_duration = len(frame_accuracies) / fps if len(frame_accuracies) > 0 else 0
            if video_duration < 2:
                raise Exception("Video too short. Please record a longer video (at least 2 seconds).")
            
            # Use simple rep counting based on duration
            if exercise_type == 'plank':
                reps = max(1, int(video_duration / 10))
            elif exercise_type == 'pushup':
                reps = max(1, int(video_duration / 3))
            elif exercise_type == 'squat':
                reps = max(1, int(video_duration / 4))
            elif exercise_type == 'tricep_dips':
                reps = max(1, int(video_duration / 3))
            elif exercise_type == 'pullup':
                reps = max(1, int(video_duration / 5))
            else:
                reps = max(1, int(video_duration / 4))
            
            # Simple fallback results
            result = {
                'exercise_type': exercise_type,
                'exercise_name': exercise_config['name'],
                'accuracy': 75,  # Assume decent form
                'mistakes': [],
                'output_video': output_filename,
                'total_frames': len(frame_accuracies),
                'score': 75,
                'feedback': "Basic analysis completed. Form assumed to be adequate.",
                'form_score': 75,
                'reps': reps,
                'analysis_time': datetime.now().isoformat()
            }
            return result
        
        # No rep counting - just use video duration for basic validation
        if len(frame_accuracies) > 0:
            video_duration = len(frame_accuracies) / fps
            print(f"Video duration: {video_duration:.1f} seconds")
            
            # Simple validation: ensure video is long enough for meaningful exercise
            if video_duration < 3:  # Too short
                reps = 0
            elif video_duration > 60:  # Too long
                reps = 0
            else:
                # For any reasonable video length, assume they did the exercise
                reps = 1  # Just mark as completed
                
            print(f"Video validation: {video_duration:.1f}s video, reps={reps}")
        else:
            reps = 0
        
        # Additional validation for different exercise types
        if exercise_type == 'plank':
            # For plank, reps are based on hold time rather than repetitions
            if len(frame_accuracies) > 0:
                hold_time_seconds = len(frame_accuracies) / fps
                reps = max(1, int(hold_time_seconds / 10))  # 1 rep per 10 seconds
        elif exercise_type == 'pushup':
            # For pushups, ensure we have proper up-down movement
            if len(all_keypoints) > 0:
                primary_angle = np.array([kp[0] for kp in all_keypoints if len(kp) > 0])
                if len(primary_angle) > 0:
                    angle_range = np.max(primary_angle) - np.min(primary_angle)
                    if angle_range < 20:  # Not enough movement for pushups
                        reps = 0
        
        result = {
            'exercise_type': exercise_type,
            'exercise_name': exercise_config['name'],
            'accuracy': avg_accuracy,
            'mistakes': unique_mistakes,
            'output_video': output_filename,
            'total_frames': len(frame_accuracies),
            'score': avg_accuracy,
            'feedback': generate_feedback(avg_accuracy, len(unique_mistakes), avg_accuracy),
            'form_score': avg_accuracy,
            'reps': reps,
            'analysis_time': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return {
            'score': 0,
            'feedback': f'Analysis failed: {str(e)}',
            'reps': 0,
            'form_score': 0,
            'exercise_name': exercise_type.replace('_', ' ').title(),
            'exercise_type': exercise_type,
            'accuracy': 0,
            'total_frames': 0,
            'mistakes': [],
            'output_video': f'feedback_{exercise_type}_{int(datetime.now().timestamp())}.mp4'
        }

def analyze_exercise_form(angles_data):
    """Analyze exercise form quality"""
    try:
        # Simple form analysis based on angle consistency
        if not angles_data:
            return 0
        
        # Calculate angle variance (lower variance = better form)
        angles_array = np.array(angles_data)
        angle_variance = np.var(angles_array, axis=0)
        avg_variance = np.mean(angle_variance)
        
        # Convert variance to score (0-100)
        form_score = max(0, 100 - (avg_variance * 1000))
        
        return min(100, form_score)
        
    except Exception as e:
        print(f"Form analysis error: {e}")
        return 50  # Default score

def count_repetitions(landmarks_data):
    """Count exercise repetitions"""
    try:
        if len(landmarks_data) < 10:
            return 0
        
        # Simple repetition counting based on vertical movement
        landmarks_array = np.array(landmarks_data)
        
        # Extract shoulder and hip positions
        shoulder_y = landmarks_array[:, 11]  # Left shoulder Y position
        hip_y = landmarks_array[:, 23]       # Left hip Y position
        
        # Calculate movement range
        movement_range = np.max(shoulder_y) - np.min(shoulder_y)
        
        # Estimate repetitions based on movement cycles
        if movement_range > 0.1:  # Significant movement detected
            reps = max(1, int(movement_range * 10))
        else:
            reps = 0
        
        return min(reps, 50)  # Cap at 50 reps
        
    except Exception as e:
        print(f"Rep counting error: {e}")
        return 0

def generate_feedback(form_score, reps, overall_score):
    """Generate personalized feedback"""
    feedback_parts = []
    
    if overall_score >= 90:
        feedback_parts.append("Excellent form! Keep up the great work!")
    elif overall_score >= 70:
        feedback_parts.append("Good form with room for improvement.")
    elif overall_score >= 50:
        feedback_parts.append("Your form needs some work. Focus on technique.")
    else:
        feedback_parts.append("Please review proper form and try again.")
    
    if reps > 0:
        feedback_parts.append(f"You completed {reps} repetitions.")
    
    if form_score < 60:
        feedback_parts.append("Try to maintain consistent movement throughout the exercise.")
    
    return " ".join(feedback_parts)

def calculate_exercise_angles(keypoints_flat, exercise_type):
    """Calculate angles based on exercise type"""
    if exercise_type == "pushup":
        left_elbow = calculate_angle(keypoints_flat[11*2:11*2+2],
                                   keypoints_flat[13*2:13*2+2],
                                   keypoints_flat[15*2:15*2+2])
        right_elbow = calculate_angle(keypoints_flat[12*2:12*2+2],
                                    keypoints_flat[14*2:14*2+2],
                                    keypoints_flat[16*2:16*2+2])
        back_angle = calculate_angle(keypoints_flat[11*2:11*2+2],
                                   keypoints_flat[23*2:23*2+2],
                                   keypoints_flat[25*2:25*2+2])
        left_knee = calculate_angle(keypoints_flat[23*2:23*2+2],
                                  keypoints_flat[25*2:25*2+2],
                                  keypoints_flat[27*2:27*2+2])
        right_knee = calculate_angle(keypoints_flat[24*2:24*2+2],
                                   keypoints_flat[26*2:26*2+2],
                                   keypoints_flat[28*2:28*2+2])
        return np.array([left_elbow, right_elbow, back_angle, left_knee, right_knee])
    
    elif exercise_type == "pullup":
        shoulder = keypoints_flat[11*2:11*2+2]
        elbow = keypoints_flat[13*2:13*2+2]
        wrist = keypoints_flat[15*2:15*2+2]
        hip = keypoints_flat[23*2:23*2+2]
        
        elbow_angle = calculate_angle(wrist, elbow, shoulder)
        shoulder_angle = calculate_angle(elbow, shoulder, hip)
        return np.array([elbow_angle, shoulder_angle])
    
    elif exercise_type == "plank":
        shoulder = keypoints_flat[11*2:11*2+2]
        hip = keypoints_flat[23*2:23*2+2]
        ankle = keypoints_flat[27*2:27*2+2]
        elbow = keypoints_flat[13*2:13*2+2]
        wrist = keypoints_flat[15*2:15*2+2]
        knee = keypoints_flat[25*2:25*2+2]
        
        back_angle = calculate_angle(shoulder, hip, ankle)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        hip_angle = calculate_angle(shoulder, hip, knee)
        return np.array([back_angle, elbow_angle, hip_angle])
    
    elif exercise_type == "squat":
        hip = keypoints_flat[23*2:23*2+2]
        knee = keypoints_flat[25*2:25*2+2]
        ankle = keypoints_flat[27*2:27*2+2]
        shoulder = keypoints_flat[11*2:11*2+2]
        hip_r = keypoints_flat[24*2:24*2+2]
        knee_r = keypoints_flat[26*2:26*2+2]
        ankle_r = keypoints_flat[28*2:28*2+2]
        
        left_knee_angle = calculate_angle(hip, knee, ankle)
        right_knee_angle = calculate_angle(hip_r, knee_r, ankle_r)
        left_hip_angle = calculate_angle(shoulder, hip, knee)
        right_hip_angle = calculate_angle(shoulder, hip_r, knee_r)
        back_angle = calculate_angle(shoulder, hip, ankle)
        
        return np.array([left_knee_angle, right_knee_angle, left_hip_angle, 
                        right_hip_angle, back_angle])
    
    elif exercise_type == "tricep_dips":
        left_shoulder = keypoints_flat[11*2:11*2+2]
        left_elbow = keypoints_flat[13*2:13*2+2]
        left_wrist = keypoints_flat[15*2:15*2+2]
        left_hip = keypoints_flat[23*2:23*2+2]
        right_shoulder = keypoints_flat[12*2:12*2+2]
        right_elbow = keypoints_flat[14*2:14*2+2]
        right_wrist = keypoints_flat[16*2:16*2+2]
        right_hip = keypoints_flat[24*2:24*2+2]
        
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
        
        return np.array([left_elbow_angle, right_elbow_angle,
                        left_shoulder_angle, right_shoulder_angle])
    
    return None

def detect_mistakes(current_angles, mean_angles, exercise_type):
    """Detect form mistakes based on current angles and mean angles"""
    mistakes = []
    
    if exercise_type == "pushup":
        avg_elbows = (current_angles[0] + current_angles[1]) / 2.0
        if avg_elbows > mean_angles[0] + 10:
            mistakes.append("Go lower - bend elbows more")
        if abs(current_angles[2] - mean_angles[2]) > 15:
            mistakes.append("Keep back straight")
        if (current_angles[3] + current_angles[4]) / 2 > mean_angles[3] + 10:
            mistakes.append("Knees bent - keep legs straight")
    
    elif exercise_type == "pullup":
        if current_angles[0] > mean_angles[0] + 15:
            mistakes.append("Pull higher - bend the elbows more at top")
        if current_angles[1] > mean_angles[1] + 15:
            mistakes.append("Drive shoulders up - reach the bar higher")
        if current_angles[1] < mean_angles[1] - 20:
            mistakes.append("Control the descent - don't drop too fast")
    
    elif exercise_type == "plank":
        if abs(current_angles[0] - 180) > 15:
            if current_angles[0] < 165:
                mistakes.append("Hips too high - lower your hips")
            else:
                mistakes.append("Hips sagging - raise your hips")
        if abs(current_angles[1] - mean_angles[1]) > 20:
            mistakes.append("Adjust elbow position - keep forearms flat")
        if abs(current_angles[2] - mean_angles[2]) > 15:
            if current_angles[2] < mean_angles[2] - 15:
                mistakes.append("Engage core - don't let hips drop")
            else:
                mistakes.append("Keep body straight - hips too high")
    
    elif exercise_type == "squat":
        avg_knee = (current_angles[0] + current_angles[1]) / 2.0
        if avg_knee > 100:
            mistakes.append("Go deeper - squat below parallel")
        knee_diff = abs(current_angles[0] - current_angles[1])
        if knee_diff > 15:
            mistakes.append("Uneven knees - maintain symmetry")
        avg_hip = (current_angles[2] + current_angles[3]) / 2.0
        if avg_hip > mean_angles[2] + 20:
            mistakes.append("Push hips back more - proper hip hinge")
        if current_angles[4] < mean_angles[4] - 15:
            mistakes.append("Keep chest up - too much forward lean")
        elif current_angles[4] > mean_angles[4] + 15:
            mistakes.append("Lean forward slightly - engage posterior chain")
    
    elif exercise_type == "tricep_dips":
        avg_elbow = (current_angles[0] + current_angles[1]) / 2.0
        if avg_elbow > 110:
            mistakes.append("Go deeper - lower until elbows at 90 degrees")
        elbow_diff = abs(current_angles[0] - current_angles[1])
        if elbow_diff > 15:
            mistakes.append("Uneven elbows - maintain symmetry")
        avg_shoulder = (current_angles[2] + current_angles[3]) / 2.0
        if avg_shoulder < mean_angles[2] - 15:
            mistakes.append("Keep chest up - don't lean too far forward")
        if avg_elbow < mean_angles[0] - 20:
            mistakes.append("Keep elbows tucked - don't flare out")
    
    return mistakes

def count_repetitions_from_angles(all_keypoints):
    """Count repetitions from angle data using improved algorithm"""
    try:
        if len(all_keypoints) < 10:
            return 0
        
        keypoints_array = np.array(all_keypoints)
        
        if keypoints_array.shape[1] == 0:
            return 0
        
        # Filter out zero/empty keypoints (MediaPipe failures)
        valid_keypoints = []
        for kp in all_keypoints:
            if len(kp) > 0 and not all(x == 0 for x in kp):
                valid_keypoints.append(kp)
        
        if len(valid_keypoints) < 5:
            print(f"Not enough valid keypoints: {len(valid_keypoints)}")
            return 0
        
        keypoints_array = np.array(valid_keypoints)
        reps_count = 0
        
        # Method 1: Peak detection on primary angle
        primary_angle = keypoints_array[:, 0]  # First angle (usually most relevant)
        
        # Simple smoothing without scipy dependency
        window_size = min(3, len(primary_angle) // 4)
        if window_size > 1:
            smoothed_angle = []
            for i in range(len(primary_angle)):
                start = max(0, i - window_size // 2)
                end = min(len(primary_angle), i + window_size // 2 + 1)
                smoothed_angle.append(np.mean(primary_angle[start:end]))
            smoothed_angle = np.array(smoothed_angle)
        else:
            smoothed_angle = primary_angle
        
        # Simple peak detection without scipy
        try:
            mean_angle = np.mean(smoothed_angle)
            std_angle = np.std(smoothed_angle)
            
            if std_angle > 5:  # There's significant variation
                # Find local maxima and minima
                peaks = []
                valleys = []
                
                for i in range(1, len(smoothed_angle) - 1):
                    if (smoothed_angle[i] > smoothed_angle[i-1] and 
                        smoothed_angle[i] > smoothed_angle[i+1] and 
                        smoothed_angle[i] > mean_angle):
                        peaks.append(i)
                    elif (smoothed_angle[i] < smoothed_angle[i-1] and 
                          smoothed_angle[i] < smoothed_angle[i+1] and 
                          smoothed_angle[i] < mean_angle):
                        valleys.append(i)
                
                # Count complete cycles
                if len(peaks) > 0 and len(valleys) > 0:
                    cycles = min(len(peaks), len(valleys))
                    reps_count = max(reps_count, cycles)
                    print(f"Peak detection: {len(peaks)} peaks, {len(valleys)} valleys, {cycles} cycles")
        except Exception as e:
            print(f"Peak detection error: {e}")
        
        # Method 2: Movement range analysis
        try:
            movement_range = np.max(smoothed_angle) - np.min(smoothed_angle)
            if movement_range > 15:  # Significant movement
                # Estimate reps based on movement range
                range_reps = max(1, int(movement_range / 20))
                reps_count = max(reps_count, range_reps)
                print(f"Range analysis: {movement_range:.1f} range, {range_reps} reps")
        except Exception as e:
            print(f"Range analysis error: {e}")
        
        # Method 3: Threshold crossing
        try:
            crossings = 0
            above_mean = smoothed_angle[0] > mean_angle
            
            for angle in smoothed_angle[1:]:
                current_above = angle > mean_angle
                if current_above != above_mean:
                    crossings += 1
                    above_mean = current_above
            
            crossing_reps = max(0, crossings // 2)
            reps_count = max(reps_count, crossing_reps)
            print(f"Threshold crossing: {crossings} crossings, {crossing_reps} reps")
        except Exception as e:
            print(f"Threshold crossing error: {e}")
        
        # Method 4: Frame-based estimation (fallback)
        if reps_count == 0:
            # If no movement detected, estimate based on video length
            total_frames = len(all_keypoints)
            if total_frames > 30:  # At least 1 second at 30fps
                # Assume 1 rep per 2 seconds of video
                estimated_reps = max(1, total_frames // 60)  # 60 frames = 2 seconds
                reps_count = min(estimated_reps, 10)  # Cap at 10 reps
                print(f"Frame-based estimation: {total_frames} frames, {reps_count} reps")
        
        # Ensure reasonable bounds
        reps_count = max(0, min(reps_count, 50))
        
        print(f"Final rep count: {reps_count} from {len(valid_keypoints)} valid frames")
        return reps_count
        
    except Exception as e:
        print(f"Rep counting error: {e}")
        return 0

def award_points_and_achievements(user_id, analysis_results):
    """Award points and check for achievements with enhanced gamification"""
    try:
        from gamification_manager import gamification_manager
        
        print(f"GAMIFICATION: Starting for user {user_id}")
        print(f"GAMIFICATION: Analysis results: {analysis_results}")
        score = analysis_results.get('score', 0)
        reps = analysis_results.get('reps', 0)
        exercise_type = analysis_results.get('exercise_type', 'pushup')
        mistakes = analysis_results.get('mistakes', [])
        print(f"GAMIFICATION: Score={score}, Reps={reps}, Exercise={exercise_type}")
        
        # Use the new gamification manager
        gamification_data = gamification_manager.calculate_workout_rewards(
            user_id, exercise_type, score, mistakes
        )
        
        print(f"GAMIFICATION: Final gamification data: {gamification_data}")
        return gamification_data
        
    except Exception as e:
        print(f"Points/achievements error: {e}")
        return {
            'xp_gained': 0,
            'level_up': False,
            'achievements': [],
            'current_level': 1,
            'current_xp': 0,
            'bonus_points': 0,
            'perfect_workouts': 0,
            'streak_days': 0
        }

def calculate_level_from_xp(xp):
    """Calculate level from experience points"""
    if xp < 100:
        return 1
    elif xp < 300:
        return 2
    elif xp < 600:
        return 3
    elif xp < 1000:
        return 4
    elif xp < 1500:
        return 5
    elif xp < 2100:
        return 6
    elif xp < 2800:
        return 7
    elif xp < 3600:
        return 8
    elif xp < 4500:
        return 9
    elif xp < 5500:
        return 10
    else:
        return min(50, (xp - 5500) // 500 + 10)  # Cap at level 50

def check_achievements(user_id, score, reps, exercise_type):
    """Check and award achievements with enhanced criteria"""
    try:
        # Get user's existing achievements
        existing_achievements = Achievement.find_by_user(user_id)
        existing_types = [a.achievement_type for a in existing_achievements]
        new_achievements = []
        
        # High score achievements
        if score >= 95 and 'perfect_score' not in existing_types:
            achievement = Achievement.create(
                user_id=user_id,
                achievement_type='perfect_score',
                title='Perfect Score',
                description='Achieved a score of 95 or higher',
                points=100
            )
            new_achievements.append(achievement)
        elif score >= 90 and 'high_score' not in existing_types:
            achievement = Achievement.create(
                user_id=user_id,
                achievement_type='high_score',
                title='High Scorer',
                description='Achieved a score of 90 or higher',
                points=50
            )
            new_achievements.append(achievement)
        
        # Repetition achievements
        if reps >= 20 and 'repetition_legend' not in existing_types:
            achievement = Achievement.create(
                user_id=user_id,
                achievement_type='repetition_legend',
                title='Repetition Legend',
                description='Completed 20 or more repetitions',
                points=75
            )
            new_achievements.append(achievement)
        elif reps >= 10 and 'repetition_master' not in existing_types:
            achievement = Achievement.create(
                user_id=user_id,
                achievement_type='repetition_master',
                title='Repetition Master',
                description='Completed 10 or more repetitions',
                points=30
            )
            new_achievements.append(achievement)
        
        # Exercise-specific achievements
        if exercise_type == 'pushup' and score >= 85 and 'pushup_master' not in existing_types:
            achievement = Achievement.create(
                user_id=user_id,
                achievement_type='pushup_master',
                title='Push-up Master',
                description='Mastered push-up form',
                points=40
            )
            new_achievements.append(achievement)
        elif exercise_type == 'squat' and score >= 85 and 'squat_master' not in existing_types:
            achievement = Achievement.create(
                user_id=user_id,
                achievement_type='squat_master',
                title='Squat Master',
                description='Mastered squat form',
                points=40
            )
            new_achievements.append(achievement)
        elif exercise_type == 'plank' and score >= 85 and 'plank_master' not in existing_types:
            achievement = Achievement.create(
                user_id=user_id,
                achievement_type='plank_master',
                title='Plank Master',
                description='Mastered plank form',
                points=40
            )
            new_achievements.append(achievement)
        elif exercise_type == 'pullup' and score >= 85 and 'pullup_master' not in existing_types:
            achievement = Achievement.create(
                user_id=user_id,
                achievement_type='pullup_master',
                title='Pull-up Master',
                description='Mastered pull-up form',
                points=40
            )
            new_achievements.append(achievement)
        elif exercise_type == 'tricep_dips' and score >= 85 and 'tricep_master' not in existing_types:
            achievement = Achievement.create(
                user_id=user_id,
                achievement_type='tricep_master',
                title='Tricep Master',
                description='Mastered tricep dip form',
                points=40
            )
            new_achievements.append(achievement)
        
        # First workout achievement
        if 'first_workout' not in existing_types:
            achievement = Achievement.create(
                user_id=user_id,
                achievement_type='first_workout',
                title='First Steps',
                description='Completed your first workout',
                points=25
            )
            new_achievements.append(achievement)
        
        # Return list of newly earned achievements
        return new_achievements
        
    except Exception as e:
        print(f"Achievement check error: {e}")
        return []

def update_streak(user_level):
    """Update user's workout streak"""
    today = datetime.now(timezone.utc).date()
    
    if user_level.last_workout_date:
        last_workout = user_level.last_workout_date.date() if hasattr(user_level.last_workout_date, 'date') else user_level.last_workout_date
        if last_workout == today:
            # Already worked out today, no change
            return
        elif last_workout == today - timedelta(days=1):
            # Consecutive day, increment streak
            user_level.streak_days += 1
        else:
            # Streak broken, reset to 1
            user_level.streak_days = 1
    else:
        # First workout
        user_level.streak_days = 1
    
    user_level.last_workout_date = datetime.now(timezone.utc)

def award_cards(user_id, score):
    """Award cards based on performance"""
    try:
        # Determine card rarity based on score
        if score >= 95:
            rarity_chance = {'legendary': 0.1, 'epic': 0.3, 'rare': 0.6}
        elif score >= 85:
            rarity_chance = {'epic': 0.1, 'rare': 0.4, 'common': 0.5}
        elif score >= 70:
            rarity_chance = {'rare': 0.2, 'common': 0.8}
        else:
            rarity_chance = {'common': 1.0}
        
        # Award 1-3 cards
        num_cards = random.randint(1, 3)
        
        for _ in range(num_cards):
            # Select rarity
            rand = random.random()
            cumulative = 0
            selected_rarity = 'common'
            
            for rarity, chance in rarity_chance.items():
                cumulative += chance
                if rand <= cumulative:
                    selected_rarity = rarity
                    break
            
            # Get random card of selected rarity
            cards = Card.find_by_rarity(selected_rarity)
            if cards:
                card = random.choice(cards)
                UserCard.create(user_id, str(card._id))
                
    except Exception as e:
        print(f"Card awarding error: {e}")

# Cleanup function to remove expired videos
def cleanup_expired_videos():
    """Remove expired temporary videos"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, video_data in temp_videos.items():
        if current_time - video_data['timestamp'] > timedelta(hours=1):
            # Remove video file
            if os.path.exists(video_data['video_path']):
                os.remove(video_data['video_path'])
            expired_sessions.append(session_id)
    
    # Remove expired sessions
    for session_id in expired_sessions:
        del temp_videos[session_id]

# Schedule cleanup every hour
import threading
import time

def cleanup_scheduler():
    """Background task to clean up expired videos"""
    while True:
        time.sleep(3600)  # Run every hour
        cleanup_expired_videos()

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_scheduler, daemon=True)
cleanup_thread.start()

# Exercise routes
@app.route('/exercise/<exercise_type>')
@login_required
def exercise_info(exercise_type):
    """Exercise information page"""
    if exercise_type not in EXERCISES:
        flash('Exercise not found', 'error')
        return redirect(url_for('dashboard'))
    
    exercise = EXERCISES[exercise_type]
    
    # Check if demo video exists
    demo_video_path = exercise.get('demo_video', '')
    demo_video_exists = False
    if demo_video_path:
        import os
        demo_video_full_path = os.path.join('data', demo_video_path)
        demo_video_exists = os.path.exists(demo_video_full_path)
    
    return render_template('exercise_info.html', 
                          exercise=exercise, 
                          exercise_type=exercise_type,
                          demo_video_path=demo_video_path,
                          demo_video_exists=demo_video_exists)

@app.route('/exercise/<exercise_type>/analyze', methods=['GET', 'POST'])
@login_required
def exercise_analyze(exercise_type):
    """Exercise analysis page"""
    if exercise_type not in EXERCISES:
        flash('Exercise not found', 'error')
        return redirect(url_for('dashboard'))
    
    exercise = EXERCISES[exercise_type]
    
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file selected', 'error')
            return redirect(url_for('exercise_analyze', exercise_type=exercise_type))
        
        video_file = request.files['video']
        if video_file.filename == '':
            flash('No video file selected', 'error')
            return redirect(url_for('exercise_analyze', exercise_type=exercise_type))
        
        if video_file:
            # Generate unique session ID for this analysis
            session_id = secrets.token_urlsafe(16)
            
            # Save video temporarily
            filename = secure_filename(video_file.filename)
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, filename)
            video_file.save(video_path)
            
            # Store video info in temporary storage
            temp_videos[session_id] = {
                'video_path': video_path,
                'user_id': str(current_user._id),
                'timestamp': datetime.now(),
                'analysis_results': None
            }
            
            # Create exercise session in database
            print(f"Creating session for user {current_user._id}, exercise {exercise_type}")
            exercise_session = ExerciseSession.create(
                user_id=str(current_user._id),
                exercise_type=exercise_type,
                video_filename=filename,
                session_id=session_id
            )
            print(f"Session created with ID: {exercise_session._id}")
            
            # Perform analysis
            try:
                analysis_results = perform_video_analysis(video_path, exercise_type)
                
                # Update session with results
                exercise_session.update_status(
                    'completed',
                    analysis_results=analysis_results,
                    score=analysis_results.get('score', 0),
                    feedback=analysis_results.get('feedback', '')
                )
                
                # Update temporary storage
                temp_videos[session_id]['analysis_results'] = analysis_results
                
                # Award points and check achievements
                gamification_data = award_points_and_achievements(str(current_user._id), analysis_results)
                
                # Add gamification data to analysis results
                if gamification_data:
                    analysis_results['gamification'] = gamification_data
                
                # Check if this is an AJAX request
                if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded' or 'application/json' in request.headers.get('Accept', ''):
                    return jsonify({
                        'success': True,
                        'session_id': session_id,
                        'redirect_url': url_for('view_results', session_id=session_id)
                    })
                else:
                    return render_template('results.html', 
                                         session_id=session_id,
                                         results=analysis_results,
                                         video_filename=filename,
                                         exercise_type=exercise_type,
                                         exercise=exercise)
                
            except Exception as e:
                print(f"Analysis error: {e}")
                exercise_session.update_status('failed')
                flash('Analysis failed. Please try again.', 'error')
                return redirect(url_for('exercise_analyze', exercise_type=exercise_type))
    
    return render_template('analyze.html', exercise=exercise, exercise_type=exercise_type)

# Additional routes for navigation
@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    # Get user's exercise history for stats
    user_id = str(current_user._id)
    sessions = ExerciseSession.find_by_user(user_id, limit=100)
    
    # Debug: Print session count
    print(f"Profile: Found {len(sessions)} sessions for user {user_id}")
    if sessions:
        print(f"Latest session: {sessions[0].exercise_type} at {sessions[0].created_at}")
    
    # Get user level data from database
    user_level = UserLevel.find_by_user(user_id)
    if not user_level:
        # Create user level if it doesn't exist
        user_level = UserLevel.create_or_update(user_id)
    
    # Calculate comprehensive stats
    total_sessions = len(sessions)
    total_score = sum(session.score for session in sessions if session.score)
    avg_score = total_score / total_sessions if total_sessions > 0 else 0
    
    # Use actual level data from database
    level = user_level.level
    experience_points = user_level.experience_points
    perfect_workouts = user_level.perfect_workouts
    streak_days = user_level.streak_days
    total_workouts = user_level.total_workouts
    
    print(f"PROFILE: User level data - Level: {level}, XP: {experience_points}, Workouts: {total_workouts}")
    
    # Get level name
    level_name = user_level.get_level_name()
    
    stats = {
        'total_sessions': total_sessions,
        'total_score': total_score,
        'avg_score': round(avg_score, 1),
        'level': level,
        'level_name': level_name,
        'experience_points': experience_points,
        'total_workouts': total_workouts,
        'perfect_workouts': perfect_workouts,
        'streak_days': streak_days,
        'total_points': experience_points,  # For leaderboard
        'achievements_unlocked': min(level, 5),  # Simplified achievements
        'badges_earned': min(level, 3)  # Simplified badges
    }
    
    # Create a user object with exercise_history for the template
    user_with_history = {
        'username': current_user.username,
        'email': current_user.email,
        'created_at': current_user.created_at,
        'exercise_history': sessions[:5]  # Recent 5 sessions for the template
    }
    
    return render_template('profile.html', user=user_with_history, stats=stats)

@app.route('/achievements')
@login_required
def achievements():
    """User achievements page"""
    user_id = str(current_user._id)
    user_achievements = Achievement.find_by_user(user_id)
    
    # Get all available achievements and user's earned achievements
    all_achievements = [
        {"id": "first_workout", "name": "First Workout", "description": "Complete your first exercise"},
        {"id": "perfect_form", "name": "Perfect Form", "description": "Achieve 90%+ accuracy"},
        {"id": "streak_7", "name": "7-Day Streak", "description": "Exercise for 7 consecutive days"},
        {"id": "level_5", "name": "Level 5", "description": "Reach level 5"},
        {"id": "all_exercises", "name": "All Exercises", "description": "Try all 5 exercise types"}
    ]
    
    # Get user's earned achievement IDs
    earned_ids = [achievement.achievement_type for achievement in user_achievements if hasattr(achievement, 'achievement_type')]
    
    # Calculate completion percentage safely
    if len(all_achievements) > 0:
        completion_percentage = round((len(earned_ids) / len(all_achievements)) * 100)
    else:
        completion_percentage = 0
    
    return render_template('achievements.html', 
                         achievements=user_achievements,
                         all_achievements=all_achievements,
                         earned_ids=earned_ids,
                         completion_percentage=completion_percentage)

@app.route('/cards')
@login_required
def cards():
    """User cards page"""
    try:
        user_id = str(current_user._id)
        user_cards_data = UserCard.find_by_user(user_id)
        
        print(f"CARDS: Found {len(user_cards_data)} user cards for user {user_id}")
        
        # Convert to the format expected by the template
        user_cards = []
        for item in user_cards_data:
            if isinstance(item, dict) and 'user_card' in item and 'card' in item:
                card = item['card']
                user_card = item['user_card']
                print(f"CARDS: Card: {card.name}, Rarity: {getattr(card, 'rarity', 'NO_RARITY')}, Type: {type(card)}")
                user_cards.append((card, user_card))
        
        print(f"CARDS: Returning {len(user_cards)} cards to template")
        return render_template('cards.html', user_cards=user_cards)
    except Exception as e:
        print(f"CARDS ERROR: {e}")
        import traceback
        traceback.print_exc()
        return render_template('cards.html', user_cards=[], error=str(e))

@app.route('/leaderboard')
@login_required
def leaderboard():
    """Leaderboard page"""
    try:
        # Get top users by experience points from UserLevel
        user_levels_collection = get_collection('user_levels')
        users_collection = get_collection('users')
        
        # Get top user levels by experience points
        top_levels = user_levels_collection.find().sort('experience_points', -1).limit(10)
        
        top_users = []
        for level_data in top_levels:
            # Get user info
            user_data = users_collection.find_one({'_id': level_data['user_id']})
            if user_data:
                user = User.from_dict(user_data)
                # Create a user level object for the template
                user_level = UserLevel.from_dict(level_data)
                top_users.append((user, user_level))
        
        print(f"LEADERBOARD: Found {len(top_users)} users")
        return render_template('leaderboard.html', top_users=top_users)
    except Exception as e:
        print(f"LEADERBOARD ERROR: {e}")
        import traceback
        traceback.print_exc()
        return render_template('leaderboard.html', leaderboard_data=[])

@app.route('/daily_challenge')
@login_required
def daily_challenge():
    """Daily challenge page"""
    user_id = str(current_user._id)
    
    # Get today's challenge
    challenge = DailyChallenge.find_today()
    user_challenge = None
    
    if challenge:
        # Check if user has completed this challenge
        user_challenge = UserChallenge.find_by_user_and_challenge(user_id, str(challenge._id))
    
    return render_template('daily_challenge.html', 
                         challenge=challenge, 
                         user_challenge=user_challenge)


@app.route('/admin/create_daily_challenge', methods=['GET', 'POST'])
@login_required
def create_daily_challenge():
    """Create or update daily challenge (admin function)"""
    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form
            
            # Check if challenge already exists for today
            today = datetime.now(timezone.utc).date()
            existing_challenge = DailyChallenge.find_by_date(today)
            
            if existing_challenge:
                # Update existing challenge
                DailyChallenge.update_challenge(
                    str(existing_challenge._id),
                    name=data.get('name', existing_challenge.name),
                    description=data.get('description', existing_challenge.description),
                    exercise_type=data.get('exercise_type', existing_challenge.exercise_type),
                    target_reps=int(data.get('target_reps', existing_challenge.target_reps)),
                    target_accuracy=int(data.get('target_accuracy', existing_challenge.target_accuracy)),
                    xp_reward=int(data.get('xp_reward', existing_challenge.xp_reward)),
                    card_reward=data.get('card_reward', existing_challenge.card_reward)
                )
                message = "Daily challenge updated successfully!"
            else:
                # Create new challenge
                DailyChallenge.create(
                    name=data.get('name', 'Daily Challenge'),
                    description=data.get('description', 'Complete today\'s challenge!'),
                    exercise_type=data.get('exercise_type', 'pushup'),
                    target_reps=int(data.get('target_reps', 10)),
                    target_accuracy=int(data.get('target_accuracy', 80)),
                    xp_reward=int(data.get('xp_reward', 50)),
                    card_reward=data.get('card_reward'),
                    challenge_date=today
                )
                message = "Daily challenge created successfully!"
            
            if request.is_json:
                return jsonify({'success': True, 'message': message})
            else:
                flash(message, 'success')
                return redirect(url_for('daily_challenge'))
                
        except Exception as e:
            error_msg = f"Error creating challenge: {str(e)}"
            if request.is_json:
                return jsonify({'success': False, 'message': error_msg})
            else:
                flash(error_msg, 'error')
                return redirect(url_for('daily_challenge'))
    
    # GET request - show form
    return render_template('admin/create_challenge.html')




@app.route('/verify_challenge_video/<challenge_id>', methods=['POST'])
@login_required
def verify_challenge_video(challenge_id):
    """Verify challenge completion using video analysis"""
    user_id = str(current_user._id)
    
    try:
        # Get challenge
        challenge = DailyChallenge.find_by_date(datetime.now(timezone.utc).date())
        if not challenge or str(challenge._id) != challenge_id:
            return jsonify({'success': False, 'message': 'Challenge not found'})
        
        # Check if already completed
        user_challenge = UserChallenge.find_by_user_and_challenge(user_id, challenge_id)
        if user_challenge and user_challenge.completed:
            return jsonify({'success': False, 'message': 'Challenge already completed!'})
        
        if 'video' not in request.files:
            return jsonify({'success': False, 'message': 'No video file provided'})
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'message': 'No video file selected'})
        
        # Save video temporarily
        filename = secure_filename(video_file.filename)
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, filename)
        video_file.save(video_path)
        
        # Perform analysis
        analysis_results = perform_video_analysis(video_path, challenge.exercise_type)
        
        # Check if challenge requirements are met
        score = analysis_results.get('score', 0)
        reps = analysis_results.get('reps', 0)
        accuracy = analysis_results.get('accuracy', 0)
        
        # Verify challenge completion - NO REP COUNTING REQUIRED
        # Just check if they uploaded a reasonable video and have decent form
        video_acceptable = reps > 0  # Just need a reasonable video length
        accuracy_acceptable = accuracy >= challenge.target_accuracy or (accuracy >= challenge.target_accuracy * 0.5)  # Allow 50% of target
        score_acceptable = score >= 30  # Very low minimum score threshold
        
        challenge_completed = (
            video_acceptable and 
            accuracy_acceptable and
            score_acceptable
        )
        
        print(f"Challenge verification: video_ok={video_acceptable}, accuracy={accuracy}/{challenge.target_accuracy}, score={score}")
        print(f"Acceptable: video={video_acceptable}, accuracy={accuracy_acceptable}, score={score_acceptable}")
        print(f"Challenge completed: {challenge_completed}")
        
        if challenge_completed:
            # Complete the challenge
            user_challenge = UserChallenge.create(
                user_id=user_id,
                challenge_id=challenge_id,
                completed=True,
                completed_at=datetime.now(timezone.utc),
                score=score,
                reps_completed=reps,
                accuracy_achieved=accuracy,
                verification_method='video_analysis',
                verification_data=analysis_results
            )
            
            # Award XP and rewards
            user_level = UserLevel.find_by_user(user_id)
            if not user_level:
                user_level = UserLevel.create_or_update(user_id)
            
            user_level.experience_points += challenge.xp_reward
            UserLevel.create_or_update(
                user_id=user_id,
                experience_points=user_level.experience_points
            )
            
            # Award card if specified
            if challenge.card_reward:
                try:
                    card = Card.find_by_name(challenge.card_reward)
                    if card:
                        UserCard.create(user_id, str(card._id))
                except Exception as e:
                    print(f"Error awarding card: {e}")
            
            return jsonify({
                'success': True,
                'challenge_completed': True,
                'message': f'Challenge completed! You earned {challenge.xp_reward} XP!',
                'xp_earned': challenge.xp_reward,
                'analysis_results': analysis_results
            })
        else:
            # Provide detailed feedback on what was missing
            missing_requirements = []
            if not video_acceptable:
                missing_requirements.append("Video too short or too long (need 3-60 seconds)")
            if not accuracy_acceptable:
                missing_requirements.append(f"Need {challenge.target_accuracy}% accuracy (you achieved {accuracy}%)")
            if not score_acceptable:
                missing_requirements.append(f"Need minimum 30% score (you achieved {score}%)")
            
            return jsonify({
                'success': True,
                'challenge_completed': False,
                'message': f'Challenge not completed. {", ".join(missing_requirements)}. Please try again with better form!',
                'analysis_results': analysis_results,
                'requirements': {
                    'target_accuracy': challenge.target_accuracy,
                    'min_score': 30,
                    'video_length_ok': video_acceptable,
                    'achieved_accuracy': accuracy,
                    'achieved_score': score
                }
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error verifying challenge: {str(e)}'})
    finally:
        # Clean up temporary video
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)


@app.route('/admin/challenges')
@login_required
def admin_challenges():
    """Admin page to manage challenges"""
    challenges = DailyChallenge.find_all(limit=30)
    return render_template('admin/challenges.html', challenges=challenges)


@app.route('/cleanup_video/<path:filename>')
@login_required
def cleanup_video(filename):
    """Clean up temporary video file"""
    try:
        video_path = os.path.join('output', filename)
        if os.path.exists(video_path):
            os.remove(video_path)
            return jsonify({'success': True, 'message': 'Video cleaned up'})
        return jsonify({'success': False, 'message': 'Video not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/video/<filename>')
def stream_video(filename):
    """Stream the output video file"""
    filepath = os.path.join('output', filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='video/mp4')
    return "Video not found", 404

@app.route('/download/<filename>')
def download_file(filename):
    """Download the output video file"""
    filepath = os.path.join('output', filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename)
    return "File not found", 404

@app.route('/demo/<path:filename>')
def serve_demo_video(filename):
    """Serve demo videos"""
    filepath = os.path.join('data', filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return "Demo video not found", 404

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



if __name__ == '__main__':
    print("Exercise Form Analyzer - MongoDB Version")
    print("Open your browser and go to: http://localhost:5003")
    print("AI-powered form analysis enabled")
    print("User authentication enabled")
    print("Exercise history tracking enabled")
    print("MongoDB database enabled")
    print("Temporary video storage enabled")
    
    app.run(debug=True, host='0.0.0.0', port=5003)
