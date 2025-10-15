// Gamification JavaScript for animations and notifications

class GamificationManager {
    constructor() {
        this.notifications = [];
        this.init();
    }

    init() {
        // Listen for reward events
        this.setupRewardListeners();
        this.setupAnimations();
    }

    setupRewardListeners() {
        // Listen for reward data in results
        if (window.location.pathname.includes('/results')) {
            this.checkForRewards();
        }
    }

    checkForRewards() {
        // Check if there are rewards in the page data
        const rewardData = this.getRewardData();
        if (rewardData) {
            this.showRewards(rewardData);
        }
    }

    getRewardData() {
        // Try to get reward data from the page
        const rewardElements = document.querySelectorAll('[data-reward]');
        if (rewardElements.length > 0) {
            return Array.from(rewardElements).map(el => ({
                type: el.dataset.reward,
                data: JSON.parse(el.dataset.rewardData || '{}')
            }));
        }
        return null;
    }

    showRewards(rewards) {
        rewards.forEach((reward, index) => {
            setTimeout(() => {
                this.showRewardNotification(reward);
            }, index * 500);
        });
    }

    showRewardNotification(reward) {
        const notification = this.createNotification(reward);
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            this.removeNotification(notification);
        }, 5000);
    }

    createNotification(reward) {
        const notification = document.createElement('div');
        notification.className = `reward-notification ${reward.type}`;
        
        let content = '';
        switch (reward.type) {
            case 'xp':
                content = `
                    <div class="notification-content">
                        <i class="fas fa-star"></i>
                        <span>+${reward.data.amount} XP Earned!</span>
                    </div>
                `;
                break;
            case 'level-up':
                content = `
                    <div class="notification-content">
                        <i class="fas fa-level-up-alt"></i>
                        <div>
                            <strong>Level Up!</strong>
                            <br>You're now ${reward.data.newLevel}!
                        </div>
                    </div>
                `;
                this.showLevelUpCelebration(reward.data);
                break;
            case 'achievement':
                content = `
                    <div class="notification-content">
                        <i class="fas fa-medal"></i>
                        <div>
                            <strong>Achievement Unlocked!</strong>
                            <br>${reward.data.name}
                        </div>
                    </div>
                `;
                break;
            case 'card':
                content = `
                    <div class="notification-content">
                        <i class="fas fa-id-card"></i>
                        <div>
                            <strong>New Card!</strong>
                            <br>${reward.data.name} (${reward.data.rarity})
                        </div>
                    </div>
                `;
                break;
        }
        
        notification.innerHTML = content;
        return notification;
    }

    showLevelUpCelebration(levelData) {
        const celebration = document.createElement('div');
        celebration.className = 'level-up-celebration';
        celebration.innerHTML = `
            <div class="level-up-content">
                <div class="sparkle"></div>
                <div class="sparkle"></div>
                <div class="sparkle"></div>
                <div class="sparkle"></div>
                <h1 class="level-up-title">🎉 LEVEL UP! 🎉</h1>
                <p class="level-up-subtitle">You're now Level ${levelData.newLevel} - ${levelData.levelName}!</p>
                <button class="level-up-close" onclick="this.parentElement.parentElement.remove()">
                    Awesome!
                </button>
            </div>
        `;
        
        document.body.appendChild(celebration);
        
        // Auto remove after 8 seconds
        setTimeout(() => {
            if (celebration.parentElement) {
                celebration.remove();
            }
        }, 8000);
    }

    removeNotification(notification) {
        notification.style.animation = 'slideOutRight 0.5s ease forwards';
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 500);
    }

    setupAnimations() {
        // Add hover effects to cards
        document.addEventListener('DOMContentLoaded', () => {
            this.setupCardAnimations();
            this.setupAchievementAnimations();
            this.setupProgressAnimations();
        });
    }

    setupCardAnimations() {
        const cards = document.querySelectorAll('.card-item');
        cards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-10px) scale(1.02)';
                card.style.boxShadow = '0 20px 40px rgba(0, 255, 136, 0.2)';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0) scale(1)';
                card.style.boxShadow = 'none';
            });
        });
    }

    setupAchievementAnimations() {
        const achievements = document.querySelectorAll('.achievement-card');
        achievements.forEach(achievement => {
            if (achievement.classList.contains('earned')) {
                // Add sparkle effect to earned achievements
                const sparkle = document.createElement('div');
                sparkle.className = 'sparkle';
                sparkle.style.position = 'absolute';
                sparkle.style.top = Math.random() * 100 + '%';
                sparkle.style.left = Math.random() * 100 + '%';
                achievement.appendChild(sparkle);
            }
        });
    }

    setupProgressAnimations() {
        const progressBars = document.querySelectorAll('.xp-progress, .progress-fill');
        progressBars.forEach(bar => {
            const width = bar.style.width;
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = width;
            }, 500);
        });
    }

    // XP Gain Animation
    showXPGain(element, amount) {
        const xpElement = document.createElement('div');
        xpElement.className = 'xp-gain-animation';
        xpElement.textContent = `+${amount} XP`;
        xpElement.style.position = 'absolute';
        xpElement.style.top = '50%';
        xpElement.style.left = '50%';
        xpElement.style.transform = 'translate(-50%, -50%)';
        
        element.style.position = 'relative';
        element.appendChild(xpElement);
        
        setTimeout(() => {
            xpElement.remove();
        }, 1500);
    }

    // Card Reveal Animation
    revealCard(cardElement) {
        cardElement.style.animation = 'cardReveal 1s ease forwards';
        cardElement.style.transform = 'scale(0) rotateY(180deg)';
        
        setTimeout(() => {
            cardElement.style.transform = 'scale(1) rotateY(0deg)';
        }, 500);
    }

    // Achievement Unlock Animation
    unlockAchievement(achievementElement) {
        achievementElement.style.animation = 'achievementUnlock 1s ease forwards';
        achievementElement.style.transform = 'scale(0) rotate(-10deg)';
        
        setTimeout(() => {
            achievementElement.style.transform = 'scale(1) rotate(0deg)';
        }, 500);
    }
}

// Initialize gamification when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.gamification = new GamificationManager();
    
    // Add CSS for slide out animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        
        .notification-content {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .notification-content i {
            font-size: 1.5rem;
        }
        
        .notification-content div {
            display: flex;
            flex-direction: column;
        }
    `;
    document.head.appendChild(style);
});

// Export for use in other scripts
window.GamificationManager = GamificationManager;

