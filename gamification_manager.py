from mongodb_models import UserLevel, Achievement, Card, UserCard, Progress
from mongodb_config import get_collection
from datetime import datetime, date, timedelta
import random
import json

class GamificationManager:
    def __init__(self):
        self.users_collection = get_collection('users')
        self.user_levels_collection = get_collection('user_levels')
        self.achievements_collection = get_collection('achievements')
        self.cards_collection = get_collection('cards')
        self.user_cards_collection = get_collection('user_cards')
        self.progress_collection = get_collection('progress')
    
    def get_or_create_user_level(self, user_id):
        """Get or create user level record"""
        user_level = UserLevel.find_by_user(user_id)
        if not user_level:
            user_level = UserLevel.create_or_update(user_id)
        return user_level
    
    def calculate_workout_rewards(self, user_id, exercise_type, accuracy, mistakes=None):
        """Calculate XP and rewards for a workout"""
        print(f"GAMIFICATION: Calculating rewards for user {user_id}, exercise {exercise_type}, accuracy {accuracy}")
        
        user_level = self.get_or_create_user_level(user_id)
        
        # Base XP calculation
        base_xp = 10
        accuracy_bonus = int(accuracy * 0.5)  # Up to 50 bonus XP for perfect form
        perfect_bonus = 25 if accuracy >= 95 else 0
        
        total_xp = base_xp + accuracy_bonus + perfect_bonus
        
        print(f"GAMIFICATION: Base XP: {base_xp}, Accuracy bonus: {accuracy_bonus}, Perfect bonus: {perfect_bonus}, Total: {total_xp}")
        
        # Update user stats
        user_level.experience_points += total_xp
        user_level.total_workouts += 1
        
        if accuracy >= 95:
            user_level.perfect_workouts += 1
        
        # Check for level up
        level_up = self.check_level_up(user_level)
        
        # Check for achievements
        new_achievements = self.check_achievements(user_id, user_level, accuracy, exercise_type)
        
        # Award cards based on performance
        new_cards = self.award_cards(user_id, accuracy, exercise_type)
        
        # Update streak
        self.update_streak(user_level)
        
        # Save updated level
        UserLevel.create_or_update(
            user_id=user_id,
            experience_points=user_level.experience_points,
            total_workouts=user_level.total_workouts,
            perfect_workouts=user_level.perfect_workouts,
            streak_days=user_level.streak_days,
            level=user_level.level
        )
        
        print(f"GAMIFICATION: Final result - XP: {total_xp}, Level up: {level_up}, Achievements: {len(new_achievements)}, Cards: {len(new_cards)}")
        
        return {
            'xp_gained': total_xp,
            'level_up': level_up,
            'new_achievements': new_achievements,
            'new_cards': new_cards,
            'current_level': user_level.level,
            'current_xp': user_level.experience_points,
            'next_level_xp': self.get_next_level_xp(user_level.level)
        }
    
    def check_level_up(self, user_level):
        """Check if user should level up"""
        required_xp = self.get_next_level_xp(user_level.level)
        if user_level.experience_points >= required_xp:
            old_level = user_level.level
            user_level.level += 1
            return {
                'leveled_up': True,
                'old_level': old_level,
                'new_level': user_level.level,
                'level_name': self.get_level_name(user_level.level)
            }
        return {'leveled_up': False}
    
    def get_next_level_xp(self, level):
        """Calculate XP needed for next level"""
        return level * 100  # Simple formula: level * 100 XP
    
    def get_level_name(self, level):
        """Get level name based on level"""
        if level < 5:
            return "Beginner"
        elif level < 10:
            return "Intermediate"
        elif level < 20:
            return "Advanced"
        else:
            return "Expert"
    
    def check_achievements(self, user_id, user_level, accuracy, exercise_type):
        """Check and award new achievements"""
        print(f"GAMIFICATION: Checking achievements for user {user_id}")
        new_achievements = []
        
        # Get user's existing achievements
        existing_achievements = Achievement.find_by_user(user_id)
        existing_types = [a.achievement_type for a in existing_achievements]
        
        # Define achievement checks
        achievement_checks = [
            {
                'type': 'first_workout',
                'title': 'First Workout',
                'description': 'Complete your first exercise',
                'points': 25,
                'condition': user_level.total_workouts == 1
            },
            {
                'type': 'perfect_form',
                'title': 'Perfect Form',
                'description': 'Achieve 95%+ accuracy',
                'points': 50,
                'condition': accuracy >= 95
            },
            {
                'type': 'streak_3',
                'title': '3-Day Streak',
                'description': 'Exercise for 3 consecutive days',
                'points': 75,
                'condition': user_level.streak_days >= 3
            },
            {
                'type': 'streak_7',
                'title': '7-Day Streak',
                'description': 'Exercise for 7 consecutive days',
                'points': 150,
                'condition': user_level.streak_days >= 7
            },
            {
                'type': 'workout_warrior',
                'title': 'Workout Warrior',
                'description': 'Complete 10 workouts',
                'points': 100,
                'condition': user_level.total_workouts >= 10
            },
            {
                'type': 'perfectionist',
                'title': 'Perfectionist',
                'description': 'Achieve 5 perfect workouts',
                'points': 200,
                'condition': user_level.perfect_workouts >= 5
            }
        ]
        
        for achievement_data in achievement_checks:
            if achievement_data['type'] not in existing_types and achievement_data['condition']:
                print(f"GAMIFICATION: Awarding achievement: {achievement_data['title']}")
                
                # Create achievement
                achievement = Achievement.create(
                    user_id=user_id,
                    achievement_type=achievement_data['type'],
                    title=achievement_data['title'],
                    description=achievement_data['description'],
                    points=achievement_data['points']
                )
                
                # Award XP
                user_level.experience_points += achievement_data['points']
                
                new_achievements.append({
                    'name': achievement_data['title'],
                    'description': achievement_data['description'],
                    'points': achievement_data['points']
                })
        
        return new_achievements
    
    def award_cards(self, user_id, accuracy, exercise_type):
        """Award cards based on performance"""
        print(f"GAMIFICATION: Awarding cards for accuracy {accuracy}")
        new_cards = []
        
        # Determine card rarity based on accuracy
        if accuracy >= 95:
            rarity_chance = {'legendary': 0.1, 'epic': 0.3, 'rare': 0.6}
        elif accuracy >= 85:
            rarity_chance = {'epic': 0.1, 'rare': 0.4, 'common': 0.5}
        elif accuracy >= 70:
            rarity_chance = {'rare': 0.2, 'common': 0.8}
        else:
            rarity_chance = {'common': 1.0}
        
        # Award 1-3 cards
        num_cards = random.randint(1, 1)
        print(f"GAMIFICATION: Awarding {num_cards} cards")
        
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
            
            print(f"GAMIFICATION: Selected rarity: {selected_rarity}")
            
            # Get random card of selected rarity
            cards = Card.find_by_rarity(selected_rarity)
            if cards:
                card = random.choice(cards)
                print(f"GAMIFICATION: Selected card: {card.name}")
                
                # Check if user already has this card
                user_card = UserCard.find_by_user_and_card(user_id, card._id)
                if user_card:
                    user_card.quantity += 1
                    UserCard.create(user_id, card._id, quantity=user_card.quantity)
                else:
                    UserCard.create(user_id, card._id)
                
                new_cards.append({
                    'name': card.name,
                    'description': card.description,
                    'rarity': card.rarity,
                    'category': card.category
                })
        
        return new_cards
    
    def update_streak(self, user_level):
        """Update user's workout streak"""
        today = date.today()
        
        if hasattr(user_level, 'last_workout_date') and user_level.last_workout_date:
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
        
        user_level.last_workout_date = datetime.now()
    
    def get_user_stats(self, user_id):
        """Get comprehensive user statistics"""
        user_level = self.get_or_create_user_level(user_id)
        
        # Get achievements
        achievements = Achievement.find_by_user(user_id)
        
        # Get cards
        user_cards = UserCard.find_by_user(user_id)
        
        return {
            'level': user_level.level,
            'level_name': self.get_level_name(user_level.level),
            'experience_points': user_level.experience_points,
            'next_level_xp': self.get_next_level_xp(user_level.level),
            'total_workouts': user_level.total_workouts,
            'perfect_workouts': user_level.perfect_workouts,
            'streak_days': user_level.streak_days,
            'achievements': [{
                'name': a.title,
                'description': a.description,
                'points': a.points
            } for a in achievements],
            'cards': [{
                'name': uc['card'].name,
                'description': uc['card'].description,
                'rarity': uc['card'].rarity,
                'category': uc['card'].category,
                'quantity': uc['user_card'].quantity
            } for uc in user_cards]
        }

# Initialize default achievements and cards
def initialize_gamification_data():
    """Initialize default achievements and cards"""
    print("GAMIFICATION: Initializing gamification data...")
    
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
            'name': 'Push-up Pro',
            'description': 'Master of push-ups',
            'rarity': 'common',
            'category': 'exercise'
        },
        {
            'name': 'Squat Specialist',
            'description': 'Squat technique expert',
            'rarity': 'rare',
            'category': 'exercise'
        },
        {
            'name': 'Plank Champion',
            'description': 'Core strength master',
            'rarity': 'epic',
            'category': 'exercise'
        }
    ]
    
    for card_data in cards_data:
        existing = Card.find_by_name(card_data['name'])
        if not existing:
            Card.create(**card_data)
            print(f"GAMIFICATION: Created card: {card_data['name']}")
    
    print("GAMIFICATION: Initialization complete!")

# Global gamification manager instance
gamification_manager = GamificationManager()
