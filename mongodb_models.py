"""
MongoDB Models for Exercise Analyzer
Production-ready data models using MongoDB
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from bson import ObjectId
from mongodb_config import get_collection
from flask_login import UserMixin
import hashlib
import secrets

class User(UserMixin):
    """User model for MongoDB"""
    
    def __init__(self, email: str, username: str, password_hash: str, **kwargs):
        self.email = email
        self.username = username
        self.password_hash = password_hash
        self.created_at = datetime.now(timezone.utc)
        self._is_active = kwargs.get('is_active', True)  # Store as private attribute
        self.last_login = None
        self.profile_data = kwargs.get('profile_data', {})
        self.preferences = kwargs.get('preferences', {})
        self._id = kwargs.get('_id', None)
    
    def get_id(self):
        """Required by Flask-Login"""
        return str(self._id) if self._id else None
    
    @property
    def is_active(self):
        """Required by Flask-Login"""
        return self._is_active
    
    @is_active.setter
    def is_active(self, value):
        """Allow setting is_active"""
        self._is_active = value
    
    @staticmethod
    def from_dict(user_data: dict):
        """Create User instance from dictionary"""
        user = User(
            email=user_data['email'],
            username=user_data['username'],
            password_hash=user_data['password_hash'],
            _id=user_data.get('_id'),
            is_active=user_data.get('is_active', True),
            profile_data=user_data.get('profile_data', {}),
            preferences=user_data.get('preferences', {})
        )
        user.created_at = user_data.get('created_at', datetime.now(timezone.utc))
        user.last_login = user_data.get('last_login')
        return user
    
    def to_dict(self):
        """Convert user to dictionary for MongoDB"""
        return {
            'email': self.email,
            'username': self.username,
            'password_hash': self.password_hash,
            'created_at': self.created_at,
            'is_active': self._is_active,
            'last_login': self.last_login,
            'profile_data': self.profile_data,
            'preferences': self.preferences
        }
    
    @staticmethod
    def create(email: str, username: str, password: str, **kwargs):
        """Create a new user"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        user = User(email, username, password_hash, **kwargs)
        
        # Save to database
        users_collection = get_collection('users')
        result = users_collection.insert_one(user.to_dict())
        user._id = result.inserted_id
        return user
    
    @staticmethod
    def find_by_email(email: str):
        """Find user by email"""
        users_collection = get_collection('users')
        user_data = users_collection.find_one({'email': email})
        if user_data:
            return User.from_dict(user_data)
        return None
    
    @staticmethod
    def find_by_id(user_id: str):
        """Find user by ID"""
        users_collection = get_collection('users')
        # Handle both ObjectId and string IDs
        try:
            from bson import ObjectId
            user_data = users_collection.find_one({'_id': ObjectId(user_id)})
        except:
            # Fallback for string IDs (mock database)
            user_data = users_collection.find_one({'_id': user_id})
        if user_data:
            return User.from_dict(user_data)
        return None
    
    @staticmethod
    def get_leaderboard(limit=10):
        """Get top users by points for leaderboard"""
        users_collection = get_collection('users')
        top_users = users_collection.find().sort('profile_data.total_points', -1).limit(limit)
        return [User.from_dict(user) for user in top_users]
    
    def update_last_login(self):
        """Update last login timestamp"""
        users_collection = get_collection('users')
        users_collection.update_one(
            {'_id': self._id},
            {'$set': {'last_login': datetime.now(timezone.utc)}}
        )
        self.last_login = datetime.now(timezone.utc)

class ExerciseSession:
    """Exercise session model for MongoDB"""
    
    def __init__(self, user_id: str, exercise_type: str, **kwargs):
        # Handle both real ObjectIds and mock IDs
        if isinstance(user_id, str):
            if user_id.startswith('mock_'):
                self.user_id = user_id  # Keep mock IDs as strings
            else:
                try:
                    self.user_id = ObjectId(user_id)
                except:
                    self.user_id = user_id  # Fallback to string if ObjectId fails
        else:
            self.user_id = user_id
        self.exercise_type = exercise_type
        self.created_at = datetime.now(timezone.utc)
        self.status = 'pending'  # pending, processing, completed, failed
        self.analysis_results = kwargs.get('analysis_results', {})
        self.score = kwargs.get('score', 0)
        self.feedback = kwargs.get('feedback', '')
        self.video_filename = kwargs.get('video_filename', '')
        self.session_id = secrets.token_urlsafe(16)
    
    def to_dict(self):
        """Convert session to dictionary for MongoDB"""
        return {
            'user_id': self.user_id,
            'exercise_type': self.exercise_type,
            'created_at': self.created_at,
            'status': self.status,
            'analysis_results': self.analysis_results,
            'score': self.score,
            'feedback': self.feedback,
            'video_filename': self.video_filename,
            'session_id': self.session_id
        }
    
    @staticmethod
    def create(user_id: str, exercise_type: str, **kwargs):
        """Create a new exercise session"""
        session = ExerciseSession(user_id, exercise_type, **kwargs)
        
        # Save to database
        sessions_collection = get_collection('exercise_sessions')
        result = sessions_collection.insert_one(session.to_dict())
        session._id = result.inserted_id
        return session
    
    @staticmethod
    def find_by_user(user_id: str, limit: int = 10):
        """Find sessions by user ID"""
        sessions_collection = get_collection('exercise_sessions')
        
        # Handle both real ObjectIds and mock IDs
        if user_id.startswith('mock_'):
            query = {'user_id': user_id}
        else:
            try:
                query = {'user_id': ObjectId(user_id)}
            except:
                query = {'user_id': user_id}
        
        sessions = sessions_collection.find(query).sort('created_at', -1).limit(limit)
        
        return [ExerciseSession.from_dict(session) for session in sessions]
    
    @staticmethod
    def find_by_session_id(session_id: str):
        """Find session by session ID"""
        sessions_collection = get_collection('exercise_sessions')
        session_data = sessions_collection.find_one({'session_id': session_id})
        if session_data:
            return ExerciseSession.from_dict(session_data)
        return None
    
    @staticmethod
    def from_dict(data):
        """Create session from dictionary"""
        session = ExerciseSession.__new__(ExerciseSession)
        session.__dict__.update(data)
        
        # Convert string dates back to datetime objects
        if isinstance(session.created_at, str):
            try:
                from datetime import datetime
                session.created_at = datetime.fromisoformat(session.created_at.replace('Z', '+00:00'))
            except:
                # If parsing fails, keep as string but add a fallback
                session.created_at = datetime.now(timezone.utc)
        
        return session
    
    def update_status(self, status: str, **kwargs):
        """Update session status"""
        sessions_collection = get_collection('exercise_sessions')
        update_data = {'status': status}
        update_data.update(kwargs)
        
        sessions_collection.update_one(
            {'_id': self._id},
            {'$set': update_data}
        )
        
        # Update local object
        for key, value in update_data.items():
            setattr(self, key, value)

class Achievement:
    """Achievement model for MongoDB"""
    
    def __init__(self, user_id: str, achievement_type: str, **kwargs):
        # Handle both real ObjectIds and mock IDs
        if isinstance(user_id, str):
            if user_id.startswith('mock_'):
                self.user_id = user_id  # Keep mock IDs as strings
            else:
                try:
                    self.user_id = ObjectId(user_id)
                except:
                    self.user_id = user_id  # Fallback to string if ObjectId fails
        else:
            self.user_id = user_id
        self.achievement_type = achievement_type
        self.earned_at = datetime.now(timezone.utc)
        self.title = kwargs.get('title', '')
        self.description = kwargs.get('description', '')
        self.icon = kwargs.get('icon', '')
        self.points = kwargs.get('points', 0)
    
    def to_dict(self):
        """Convert achievement to dictionary for MongoDB"""
        return {
            'user_id': self.user_id,
            'achievement_type': self.achievement_type,
            'earned_at': self.earned_at,
            'title': self.title,
            'description': self.description,
            'icon': self.icon,
            'points': self.points
        }
    
    @staticmethod
    def create(user_id: str, achievement_type: str, **kwargs):
        """Create a new achievement"""
        achievement = Achievement(user_id, achievement_type, **kwargs)
        
        # Save to database
        achievements_collection = get_collection('achievements')
        result = achievements_collection.insert_one(achievement.to_dict())
        achievement._id = result.inserted_id
        return achievement
    
    @staticmethod
    def find_by_user(user_id: str):
        """Find achievements by user ID"""
        achievements_collection = get_collection('achievements')
        
        # Handle both real ObjectIds and mock IDs
        if user_id.startswith('mock_'):
            query = {'user_id': user_id}
        else:
            try:
                query = {'user_id': ObjectId(user_id)}
            except:
                query = {'user_id': user_id}
        
        achievements = achievements_collection.find(query).sort('earned_at', -1)
        
        return [Achievement.from_dict(achievement) for achievement in achievements]
    
    @staticmethod
    def find_by_type(achievement_type: str):
        """Find achievements by type"""
        achievements_collection = get_collection('achievements')
        achievements = achievements_collection.find(
            {'achievement_type': achievement_type}
        ).sort('earned_at', -1)
        
        return [Achievement.from_dict(achievement) for achievement in achievements]
    
    @staticmethod
    def from_dict(data):
        """Create achievement from dictionary"""
        achievement = Achievement.__new__(Achievement)
        achievement.__dict__.update(data)
        return achievement

class Progress:
    """Progress tracking model for MongoDB"""
    
    def __init__(self, user_id: str, date: datetime, **kwargs):
        self.user_id = ObjectId(user_id) if isinstance(user_id, str) else user_id
        self.date = date
        self.exercises_completed = kwargs.get('exercises_completed', 0)
        self.total_score = kwargs.get('total_score', 0)
        self.workout_duration = kwargs.get('workout_duration', 0)
        self.calories_burned = kwargs.get('calories_burned', 0)
        self.achievements_earned = kwargs.get('achievements_earned', [])
    
    def to_dict(self):
        """Convert progress to dictionary for MongoDB"""
        return {
            'user_id': self.user_id,
            'date': self.date,
            'exercises_completed': self.exercises_completed,
            'total_score': self.total_score,
            'workout_duration': self.workout_duration,
            'calories_burned': self.calories_burned,
            'achievements_earned': self.achievements_earned
        }
    
    @staticmethod
    def create(user_id: str, date: datetime, **kwargs):
        """Create or update progress"""
        progress_collection = get_collection('progress')
        
        # Check if progress exists for this date
        existing = progress_collection.find_one({
            'user_id': ObjectId(user_id),
            'date': date
        })
        
        if existing:
            # Update existing progress
            progress_collection.update_one(
                {'_id': existing['_id']},
                {'$set': kwargs}
            )
            progress = Progress.from_dict({**existing, **kwargs})
        else:
            # Create new progress
            progress = Progress(user_id, date, **kwargs)
            result = progress_collection.insert_one(progress.to_dict())
            progress._id = result.inserted_id
        
        return progress
    
    @staticmethod
    def find_by_user(user_id: str, days: int = 30):
        """Find progress by user ID for last N days"""
        from datetime import timedelta
        
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        progress_collection = get_collection('progress')
        progress_data = progress_collection.find({
            'user_id': ObjectId(user_id),
            'date': {'$gte': start_date}
        }).sort('date', -1)
        
        return [Progress.from_dict(progress) for progress in progress_data]
    
    @staticmethod
    def from_dict(data):
        """Create progress from dictionary"""
        progress = Progress.__new__(Progress)
        progress.__dict__.update(data)
        return progress


class Card:
    """Card model for MongoDB"""
    
    def __init__(self, name: str, description: str, rarity: str, category: str, **kwargs):
        self.name = name
        self.description = description
        self.rarity = rarity
        self.category = category
        self.is_active = kwargs.get('is_active', True)
        self.image_url = kwargs.get('image_url', '')
        self._id = kwargs.get('_id', None)
    
    def to_dict(self):
        """Convert card to dictionary for MongoDB"""
        return {
            'name': self.name,
            'description': self.description,
            'rarity': self.rarity,
            'category': self.category,
            'is_active': self.is_active,
            'image_url': self.image_url
        }
    
    @staticmethod
    def create(name: str, description: str, rarity: str, category: str, **kwargs):
        """Create a new card"""
        card = Card(name, description, rarity, category, **kwargs)
        
        # Save to database
        cards_collection = get_collection('cards')
        result = cards_collection.insert_one(card.to_dict())
        card._id = result.inserted_id
        return card
    
    @staticmethod
    def find_by_rarity(rarity: str):
        """Find cards by rarity"""
        cards_collection = get_collection('cards')
        cards_data = cards_collection.find({'rarity': rarity, 'is_active': True})
        return [Card.from_dict(card) for card in cards_data]
    
    @staticmethod
    def find_all():
        """Find all active cards"""
        cards_collection = get_collection('cards')
        cards_data = cards_collection.find({'is_active': True})
        return [Card.from_dict(card) for card in cards_data]
    
    @staticmethod
    def find_by_name(name: str):
        """Find card by name"""
        cards_collection = get_collection('cards')
        card_data = cards_collection.find_one({'name': name})
        return Card.from_dict(card_data) if card_data else None
    
    @staticmethod
    def from_dict(data):
        """Create card from dictionary"""
        card = Card.__new__(Card)
        card.__dict__.update(data)
        return card


class UserCard:
    """User card collection model for MongoDB"""
    
    def __init__(self, user_id: str, card_id: str, **kwargs):
        # Handle both real ObjectIds and mock IDs for user_id
        if isinstance(user_id, str):
            if user_id.startswith('mock_'):
                self.user_id = user_id  # Keep mock IDs as strings
            else:
                try:
                    self.user_id = ObjectId(user_id)
                except:
                    self.user_id = user_id  # Fallback to string if ObjectId fails
        else:
            self.user_id = user_id
            
        # Handle both real ObjectIds and mock IDs for card_id
        if isinstance(card_id, str):
            if card_id.startswith('mock_'):
                self.card_id = card_id  # Keep mock IDs as strings
            else:
                try:
                    self.card_id = ObjectId(card_id)
                except:
                    self.card_id = card_id  # Fallback to string if ObjectId fails
        else:
            self.card_id = card_id
        self.quantity = kwargs.get('quantity', 1)
        self.obtained_at = kwargs.get('obtained_at', datetime.now(timezone.utc))
        self._id = kwargs.get('_id', None)
    
    def to_dict(self):
        """Convert user card to dictionary for MongoDB"""
        return {
            'user_id': self.user_id,
            'card_id': self.card_id,
            'quantity': self.quantity,
            'obtained_at': self.obtained_at
        }
    
    @staticmethod
    def create(user_id: str, card_id: str, **kwargs):
        """Create or update user card"""
        user_cards_collection = get_collection('user_cards')
        
        # Check if user already has this card
        existing = user_cards_collection.find_one({
            'user_id': ObjectId(user_id),
            'card_id': ObjectId(card_id)
        })
        
        if existing:
            # Update quantity
            user_cards_collection.update_one(
                {'_id': existing['_id']},
                {'$inc': {'quantity': 1}}
            )
            # Return updated document
            updated = user_cards_collection.find_one({'_id': existing['_id']})
            return UserCard.from_dict(updated)
        else:
            # Create new user card
            user_card = UserCard(user_id, card_id, **kwargs)
            result = user_cards_collection.insert_one(user_card.to_dict())
            user_card._id = result.inserted_id
            return user_card
    
    @staticmethod
    def find_by_user(user_id: str):
        """Find all cards for a user"""
        user_cards_collection = get_collection('user_cards')
        cards_collection = get_collection('cards')
        
        # Handle both real ObjectIds and mock IDs
        if user_id.startswith('mock_'):
            query = {'user_id': user_id}
        else:
            try:
                query = {'user_id': ObjectId(user_id)}
            except:
                query = {'user_id': user_id}
        
        # Get user's cards
        user_cards_data = user_cards_collection.find(query)
        user_cards = []
        
        for user_card_data in user_cards_data:
            # Get card details
            card_data = cards_collection.find_one({'_id': user_card_data['card_id']})
            if card_data:
                user_cards.append({
                    'user_card': UserCard.from_dict(user_card_data),
                    'card': Card.from_dict(card_data)
                })
        
        return user_cards
    
    @staticmethod
    def find_by_user_and_card(user_id: str, card_id: str):
        """Find user card by user and card ID"""
        user_cards_collection = get_collection('user_cards')
        
        # Handle both real ObjectIds and mock IDs
        if user_id.startswith('mock_'):
            user_query = {'user_id': user_id}
        else:
            try:
                user_query = {'user_id': ObjectId(user_id)}
            except:
                user_query = {'user_id': user_id}
        
        if isinstance(card_id, str) and card_id.startswith('mock_'):
            card_query = {'card_id': card_id}
        else:
            try:
                if isinstance(card_id, str):
                    card_query = {'card_id': ObjectId(card_id)}
                else:
                    card_query = {'card_id': card_id}
            except:
                card_query = {'card_id': card_id}
        
        query = {**user_query, **card_query}
        user_card_data = user_cards_collection.find_one(query)
        return UserCard.from_dict(user_card_data) if user_card_data else None
    
    @staticmethod
    def from_dict(data):
        """Create user card from dictionary"""
        user_card = UserCard.__new__(UserCard)
        user_card.__dict__.update(data)
        return user_card


class UserLevel:
    """User level and stats model for MongoDB"""
    
    def __init__(self, user_id: str, **kwargs):
        self.user_id = ObjectId(user_id) if isinstance(user_id, str) else user_id
        self.level = kwargs.get('level', 1)
        self.experience_points = kwargs.get('experience_points', 0)
        self.total_workouts = kwargs.get('total_workouts', 0)
        self.perfect_workouts = kwargs.get('perfect_workouts', 0)
        self.streak_days = kwargs.get('streak_days', 0)
        self.last_workout_date = kwargs.get('last_workout_date', None)
        self._id = kwargs.get('_id', None)
    
    def to_dict(self):
        """Convert user level to dictionary for MongoDB"""
        return {
            'user_id': self.user_id,
            'level': self.level,
            'experience_points': self.experience_points,
            'total_workouts': self.total_workouts,
            'perfect_workouts': self.perfect_workouts,
            'streak_days': self.streak_days,
            'last_workout_date': self.last_workout_date
        }
    
    def get_next_level_xp(self):
        """Calculate XP needed for next level"""
        return self.level * 100
    
    def get_level_name(self):
        """Get level name based on level"""
        if self.level < 5:
            return "Beginner"
        elif self.level < 10:
            return "Intermediate"
        elif self.level < 20:
            return "Advanced"
        else:
            return "Expert"
    
    @staticmethod
    def create_or_update(user_id: str, **kwargs):
        """Create or update user level"""
        user_levels_collection = get_collection('user_levels')
        
        # Check if user level exists
        existing = user_levels_collection.find_one({'user_id': ObjectId(user_id)})
        
        if existing:
            # Update existing
            update_data = {}
            for key, value in kwargs.items():
                update_data[key] = value
            
            user_levels_collection.update_one(
                {'user_id': ObjectId(user_id)},
                {'$set': update_data}
            )
            # Return updated document
            updated = user_levels_collection.find_one({'user_id': ObjectId(user_id)})
            return UserLevel.from_dict(updated)
        else:
            # Create new
            user_level = UserLevel(user_id, **kwargs)
            result = user_levels_collection.insert_one(user_level.to_dict())
            user_level._id = result.inserted_id
            return user_level
    
    @staticmethod
    def find_by_user(user_id: str):
        """Find user level by user ID"""
        user_levels_collection = get_collection('user_levels')
        
        # Handle both real ObjectIds and mock IDs
        if user_id.startswith('mock_'):
            query = {'user_id': user_id}
        else:
            try:
                query = {'user_id': ObjectId(user_id)}
            except:
                query = {'user_id': user_id}
        
        user_level_data = user_levels_collection.find_one(query)
        if user_level_data:
            return UserLevel.from_dict(user_level_data)
        return None
    
    @staticmethod
    def from_dict(data):
        """Create user level from dictionary"""
        user_level = UserLevel.__new__(UserLevel)
        user_level.__dict__.update(data)
        return user_level


class DailyChallenge:
    """Daily challenge model for MongoDB"""
    
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', '')
        self.description = kwargs.get('description', '')
        self.exercise_type = kwargs.get('exercise_type', 'pushup')
        self.target_reps = kwargs.get('target_reps', 10)
        self.target_accuracy = kwargs.get('target_accuracy', 80)
        self.xp_reward = kwargs.get('xp_reward', 50)
        self.card_reward = kwargs.get('card_reward', None)
        self.challenge_date = kwargs.get('challenge_date', datetime.now(timezone.utc).date())
        self.is_active = kwargs.get('is_active', True)
        self.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
        self._id = kwargs.get('_id', None)
    
    def to_dict(self):
        """Convert challenge to dictionary for MongoDB"""
        return {
            'name': self.name,
            'description': self.description,
            'exercise_type': self.exercise_type,
            'target_reps': self.target_reps,
            'target_accuracy': self.target_accuracy,
            'xp_reward': self.xp_reward,
            'card_reward': self.card_reward,
            'challenge_date': self.challenge_date.isoformat() if hasattr(self.challenge_date, 'isoformat') else str(self.challenge_date),
            'is_active': self.is_active,
            'created_at': self.created_at
        }
    
    @staticmethod
    def create(**kwargs):
        """Create a new daily challenge"""
        challenge = DailyChallenge(**kwargs)
        
        # Save to database
        challenges_collection = get_collection('daily_challenges')
        result = challenges_collection.insert_one(challenge.to_dict())
        challenge._id = result.inserted_id
        return challenge
    
    @staticmethod
    def find_by_date(challenge_date):
        """Find challenge by date"""
        challenges_collection = get_collection('daily_challenges')
        # Convert date to string for comparison
        date_str = challenge_date.isoformat() if hasattr(challenge_date, 'isoformat') else str(challenge_date)
        challenge_data = challenges_collection.find_one({
            'challenge_date': date_str,
            'is_active': True
        })
        return DailyChallenge.from_dict(challenge_data) if challenge_data else None
    
    @staticmethod
    def find_today():
        """Find today's challenge"""
        today = datetime.now(timezone.utc).date()
        return DailyChallenge.find_by_date(today)
    
    @staticmethod
    def find_all(limit=30):
        """Find all challenges"""
        challenges_collection = get_collection('daily_challenges')
        challenges_data = challenges_collection.find({'is_active': True}).sort('challenge_date', -1).limit(limit)
        return [DailyChallenge.from_dict(challenge) for challenge in challenges_data]
    
    @staticmethod
    def update_challenge(challenge_id, **kwargs):
        """Update challenge"""
        challenges_collection = get_collection('daily_challenges')
        challenges_collection.update_one(
            {'_id': ObjectId(challenge_id)},
            {'$set': kwargs}
        )
    
    @staticmethod
    def from_dict(data):
        """Create challenge from dictionary"""
        challenge = DailyChallenge.__new__(DailyChallenge)
        challenge.__dict__.update(data)
        
        # Convert string date back to date object
        if isinstance(challenge.challenge_date, str):
            try:
                from datetime import date
                challenge.challenge_date = date.fromisoformat(challenge.challenge_date)
            except:
                # If parsing fails, keep as string
                pass
        
        return challenge


class UserChallenge:
    """User challenge completion model for MongoDB"""
    
    def __init__(self, user_id: str, challenge_id: str, **kwargs):
        # Handle both real ObjectIds and mock IDs
        if isinstance(user_id, str):
            if user_id.startswith('mock_'):
                self.user_id = user_id  # Keep mock IDs as strings
            else:
                try:
                    self.user_id = ObjectId(user_id)
                except:
                    self.user_id = user_id  # Fallback to string if ObjectId fails
        else:
            self.user_id = user_id
            
        if isinstance(challenge_id, str):
            if challenge_id.startswith('mock_'):
                self.challenge_id = challenge_id  # Keep mock IDs as strings
            else:
                try:
                    self.challenge_id = ObjectId(challenge_id)
                except:
                    self.challenge_id = challenge_id  # Fallback to string if ObjectId fails
        else:
            self.challenge_id = challenge_id
            
        self.completed = kwargs.get('completed', False)
        self.completed_at = kwargs.get('completed_at', None)
        self.score = kwargs.get('score', 0)
        self.reps_completed = kwargs.get('reps_completed', 0)
        self.accuracy_achieved = kwargs.get('accuracy_achieved', 0)
        self.verification_method = kwargs.get('verification_method', 'manual')  # manual, video_analysis
        self.verification_data = kwargs.get('verification_data', {})
        self._id = kwargs.get('_id', None)
    
    def to_dict(self):
        """Convert user challenge to dictionary for MongoDB"""
        return {
            'user_id': self.user_id,
            'challenge_id': self.challenge_id,
            'completed': self.completed,
            'completed_at': self.completed_at,
            'score': self.score,
            'reps_completed': self.reps_completed,
            'accuracy_achieved': self.accuracy_achieved,
            'verification_method': self.verification_method,
            'verification_data': self.verification_data
        }
    
    @staticmethod
    def create(user_id: str, challenge_id: str, **kwargs):
        """Create or update user challenge"""
        user_challenges_collection = get_collection('user_challenges')
        
        # Check if user challenge exists
        existing = user_challenges_collection.find_one({
            'user_id': ObjectId(user_id) if not str(user_id).startswith('mock_') else user_id,
            'challenge_id': ObjectId(challenge_id) if not str(challenge_id).startswith('mock_') else challenge_id
        })
        
        if existing:
            # Update existing
            update_data = {}
            for key, value in kwargs.items():
                update_data[key] = value
            
            user_challenges_collection.update_one(
                {'_id': existing['_id']},
                {'$set': update_data}
            )
            # Return updated document
            updated = user_challenges_collection.find_one({'_id': existing['_id']})
            return UserChallenge.from_dict(updated)
        else:
            # Create new
            user_challenge = UserChallenge(user_id, challenge_id, **kwargs)
            result = user_challenges_collection.insert_one(user_challenge.to_dict())
            user_challenge._id = result.inserted_id
            return user_challenge
    
    @staticmethod
    def find_by_user_and_challenge(user_id: str, challenge_id: str):
        """Find user challenge by user and challenge ID"""
        user_challenges_collection = get_collection('user_challenges')
        
        # Handle both real ObjectIds and mock IDs
        if user_id.startswith('mock_'):
            user_query = {'user_id': user_id}
        else:
            try:
                user_query = {'user_id': ObjectId(user_id)}
            except:
                user_query = {'user_id': user_id}
        
        if isinstance(challenge_id, str) and challenge_id.startswith('mock_'):
            challenge_query = {'challenge_id': challenge_id}
        else:
            try:
                if isinstance(challenge_id, str):
                    challenge_query = {'challenge_id': ObjectId(challenge_id)}
                else:
                    challenge_query = {'challenge_id': challenge_id}
            except:
                challenge_query = {'challenge_id': challenge_id}
        
        query = {**user_query, **challenge_query}
        user_challenge_data = user_challenges_collection.find_one(query)
        return UserChallenge.from_dict(user_challenge_data) if user_challenge_data else None
    
    @staticmethod
    def find_by_user(user_id: str, limit=10):
        """Find user challenges by user ID"""
        user_challenges_collection = get_collection('user_challenges')
        
        # Handle both real ObjectIds and mock IDs
        if user_id.startswith('mock_'):
            query = {'user_id': user_id}
        else:
            try:
                query = {'user_id': ObjectId(user_id)}
            except:
                query = {'user_id': user_id}
        
        user_challenges_data = user_challenges_collection.find(query).sort('completed_at', -1).limit(limit)
        return [UserChallenge.from_dict(user_challenge) for user_challenge in user_challenges_data]
    
    @staticmethod
    def from_dict(data):
        """Create user challenge from dictionary"""
        user_challenge = UserChallenge.__new__(UserChallenge)
        user_challenge.__dict__.update(data)
        return user_challenge
