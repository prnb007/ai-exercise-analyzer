# 🏋️ AI-Powered Exercise Form Analyzer

A cutting-edge web application that uses AI and computer vision to analyze exercise form in real-time, providing personalized feedback to help users improve their workout technique.

## ✨ Features

### 🎯 Core Functionality
- **AI-Powered Form Analysis**: Advanced pose detection using MediaPipe and custom LSTM models
- **Real-time Feedback**: Instant analysis of exercise form with accuracy scoring
- **Multiple Exercise Support**: Push-ups, Pull-ups, Planks, Squats, and Tricep Dips
- **Video Processing**: Upload and analyze workout videos with detailed feedback

### 🎮 Gamification System
- **User Levels & XP**: Progressive leveling system with experience points
- **Achievements**: Unlock badges for milestones and consistency
- **Card Collection**: Collect and trade exercise-themed cards
- **Daily Challenges**: Complete daily workout challenges for rewards
- **Leaderboards**: Compete with other users globally

### 🎨 Modern UI/UX
- **Apple-style Design**: Clean, modern interface with smooth animations
- **Parallax Effects**: Beautiful scrolling animations and visual effects
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Dark Theme**: Eye-friendly dark mode with vibrant accents

### 🔐 User Management
- **Secure Authentication**: User registration and login system
- **Profile Management**: Track personal stats and progress
- **Exercise History**: View past workouts and improvements
- **Progress Tracking**: Monitor your fitness journey over time

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- MongoDB Atlas account (or local MongoDB)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd exercise_2_project_updated
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MongoDB**
   - Create a MongoDB Atlas account
   - Create a new cluster
   - Get your connection string
   - Update `mongodb_config.py` with your credentials

4. **Run the application**
   ```bash
   python app_mongodb.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:5003`
   - Start analyzing your workouts!

## 📁 Project Structure

```
exercise_2_project_updated/
├── app_mongodb.py              # Main Flask application
├── forms.py                    # WTForms for user input
├── gamification_manager.py     # Gamification system logic
├── mongodb_config.py           # Database configuration
├── mongodb_models.py           # Database models
├── requirements.txt            # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css          # Main stylesheet with animations
│   └── js/
│       ├── main.js            # Core JavaScript
│       └── gamification.js       # Gamification features
├── templates/
│   ├── base.html              # Base template
│   ├── index.html             # Homepage
│   ├── login.html             # User login
│   ├── signup.html            # User registration
│   ├── profile.html           # User profile
│   ├── analyze.html           # Video analysis
│   ├── results.html           # Analysis results
│   ├── achievements.html      # User achievements
│   ├── cards.html             # Card collection
│   ├── leaderboard.html       # Global leaderboard
│   └── daily_challenge.html   # Daily challenges
├── models/                     # AI model files
│   ├── *.npz                  # Angle statistics
│   └── *.pt                   # LSTM model weights
└── data/                      # Sample exercise videos
    ├── pushup/
    ├── pullup/
    ├── plank/
    ├── squat/
    └── tricep_dips/
```

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework
- **MongoDB**: Database for user data and analytics
- **MediaPipe**: Pose detection and analysis
- **OpenCV**: Video processing
- **NumPy**: Numerical computations
- **PyTorch**: LSTM models for form analysis

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with animations
- **JavaScript**: Interactive features and parallax effects
- **Bootstrap**: Responsive design framework

### AI/ML
- **MediaPipe Pose**: Real-time pose detection
- **Custom LSTM Models**: Exercise-specific form analysis
- **Computer Vision**: Video processing and analysis

## 🎯 Supported Exercises

### 1. Push-ups
- **Target Muscles**: Chest, Shoulders, Triceps
- **Analysis**: Range of motion, body alignment, tempo
- **Feedback**: Hand placement, core engagement, full extension

### 2. Pull-ups
- **Target Muscles**: Back, Biceps, Shoulders
- **Analysis**: Grip strength, full range of motion, controlled descent
- **Feedback**: Dead hang position, chin over bar, controlled movement

### 3. Planks
- **Target Muscles**: Core, Shoulders, Glutes
- **Analysis**: Body alignment, stability, duration
- **Feedback**: Straight line from head to heels, core engagement

### 4. Squats
- **Target Muscles**: Quadriceps, Glutes, Hamstrings
- **Analysis**: Depth, knee tracking, hip mobility
- **Feedback**: Full depth, knee alignment, weight distribution

### 5. Tricep Dips
- **Target Muscles**: Triceps, Chest, Shoulders
- **Analysis**: Range of motion, body position, control
- **Feedback**: Full extension, controlled movement, proper setup

## 🎮 Gamification Features

### Leveling System
- **XP Gained**: Earn experience points for each workout
- **Level Progression**: Unlock new features and rewards
- **Skill Trees**: Specialize in different exercise types

### Achievements
- **First Workout**: Complete your first analysis
- **Consistency**: Work out for multiple days in a row
- **Form Master**: Achieve high accuracy scores
- **Challenge Champion**: Complete daily challenges

### Card Collection
- **Rarity System**: Common, Rare, Epic, Legendary cards
- **Exercise Cards**: Collect cards for each exercise type
- **Trading System**: Exchange cards with other users

### Daily Challenges
- **Rotating Challenges**: New challenges every day
- **Reward System**: Earn XP, cards, and achievements
- **Progress Tracking**: Monitor your challenge completion

## 📊 Analytics & Insights

### Personal Analytics
- **Progress Tracking**: Monitor improvements over time
- **Form Analysis**: Detailed breakdown of technique
- **Workout History**: Complete record of all sessions
- **Performance Metrics**: Accuracy, consistency, and growth

### Social Features
- **Leaderboards**: Compare with other users globally
- **Achievement Sharing**: Show off your accomplishments
- **Community Challenges**: Participate in group events

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:

```env
MONGODB_URI=your_mongodb_connection_string
SECRET_KEY=your_secret_key
FLASK_ENV=development
```

### Database Setup
1. Create a MongoDB Atlas cluster
2. Get your connection string
3. Update `mongodb_config.py` with your credentials
4. Run the application to auto-create collections

## 🚀 Deployment

### Local Development
```bash
python app_mongodb.py
```

### Production Deployment
1. Set up a production MongoDB cluster
2. Configure environment variables
3. Use a production WSGI server (Gunicorn)
4. Set up reverse proxy (Nginx)
5. Configure SSL certificates

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe**: For pose detection capabilities
- **OpenCV**: For video processing
- **Flask**: For the web framework
- **MongoDB**: For database services
- **Community**: For feedback and contributions

## 📞 Support

If you encounter any issues or have questions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Contact the development team

## 🔮 Future Features

- **Real-time Analysis**: Live video analysis during workouts
- **Mobile App**: Native iOS and Android applications
- **Social Features**: Connect with friends and trainers
- **Advanced Analytics**: Machine learning insights
- **Integration**: Connect with fitness trackers and apps

---

**Built with ❤️ for fitness enthusiasts who want to perfect their form and achieve their goals.**

*Train. Transform. Triumph.* 🏋️‍♂️💪