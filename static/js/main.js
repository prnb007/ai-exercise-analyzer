// ===================================
// MOTIVATIONAL FITNESS WEB APP
// Apple-style Parallax & Interactions
// ===================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing app...');
    
    // Initialize core functionality
    initParallaxHero();
    initScrollAnimations();
    initNavbarBehavior();
    initCardAnimations();
    initSmoothScrolling();
    initNavigationDropdown();
    initFlashMessages();
    
    // Initialize parallax effects after a delay to ensure DOM is ready
    setTimeout(() => {
        initSectionParallax();
    }, 100);
    
    console.log('All functions initialized');
    
    // Test hamburger menu specifically
    setTimeout(() => {
        const hamburgerMenu = document.getElementById('hamburger-menu');
        const userDropdown = document.getElementById('user-dropdown');
        console.log('Hamburger menu test - Elements found:', {
            hamburger: !!hamburgerMenu,
            dropdown: !!userDropdown
        });
    }, 1000);
});

// ===================================
// PARALLAX HERO EFFECT
// ===================================
function initParallaxHero() {
    const heroContent = document.querySelector('.hero-content');
    const heroSection = document.querySelector('.hero-section');
    
    if (!heroContent || !heroSection) {
        console.log('Hero elements not found for parallax');
        return;
    }
    
    console.log('Initializing parallax hero effect');
    
    // Add a visual indicator for testing
    const testIndicator = document.createElement('div');
    testIndicator.style.cssText = `
        position: fixed;
        top: 10px;
        left: 10px;
        background: rgba(0, 255, 136, 0.8);
        color: #000;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        z-index: 10000;
        font-family: monospace;
    `;
    testIndicator.textContent = 'Parallax: Ready';
    document.body.appendChild(testIndicator);
    
    // Update indicator on scroll
    let lastScrollY = 0;
    window.addEventListener('scroll', () => {
        const scrollY = window.pageYOffset;
        if (Math.abs(scrollY - lastScrollY) > 10) {
            testIndicator.textContent = `Parallax: ${Math.round(scrollY)}px`;
            lastScrollY = scrollY;
        }
    });
    
    let ticking = false;
    
    function updateParallax() {
        const scrolled = window.pageYOffset;
        const heroHeight = heroSection.offsetHeight;
        
        // Only apply parallax within hero section
        if (scrolled < heroHeight) {
            // Smooth parallax effect - content moves slower than scroll
            const parallaxSpeed = 0.3;
            const yPos = scrolled * parallaxSpeed;
            
            // Apply transform with 3D acceleration
            heroContent.style.transform = `translate3d(0, ${yPos}px, 0)`;
            heroContent.style.willChange = 'transform';
            
            // Fade out as user scrolls
            const opacity = 1 - (scrolled / heroHeight) * 0.8;
            heroContent.style.opacity = Math.max(0.2, opacity);
            
            // Debug logging (remove in production)
            if (scrolled > 0 && scrolled % 100 === 0) {
                console.log('Parallax active:', { scrolled, yPos, opacity });
            }
        }
        
        ticking = false;
    }
    
    window.addEventListener('scroll', function() {
        if (!ticking) {
            window.requestAnimationFrame(updateParallax);
            ticking = true;
        }
    });
}

// ===================================
// SCROLL ANIMATIONS
// ===================================
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.15,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                // Stagger animation delay
                const delay = index * 100;
                
                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0) scale(1)';
                }, delay);
                
                // Stop observing after animation
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe all cards
    const cards = document.querySelectorAll('.exercise-card, .feature-card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(60px) scale(0.95)';
        card.style.transition = `all 0.8s cubic-bezier(0.4, 0, 0.2, 1) ${index * 0.1}s`;
        observer.observe(card);
    });
}

// ===================================
// NAVBAR BEHAVIOR
// ===================================
function initNavbarBehavior() {
    const navbar = document.querySelector('.navbar');
    let lastScrollTop = 0;
    let ticking = false;
    
    window.addEventListener('scroll', function() {
        if (!ticking) {
            window.requestAnimationFrame(function() {
                const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                
                // Add scrolled class for styling
                if (scrollTop > 100) {
                    navbar.classList.add('scrolled');
                } else {
                    navbar.classList.remove('scrolled');
                }
                
                // Hide/show navbar on scroll direction
                if (scrollTop > lastScrollTop && scrollTop > 300) {
                    // Scrolling down
                    navbar.style.transform = 'translateY(-100%)';
                } else {
                    // Scrolling up
                    navbar.style.transform = 'translateY(0)';
                }
                
                lastScrollTop = scrollTop <= 0 ? 0 : scrollTop;
                ticking = false;
            });
            
            ticking = true;
        }
    });
}

// ===================================
// CARD HOVER ANIMATIONS
// ===================================
function initCardAnimations() {
    const cards = document.querySelectorAll('.exercise-card');
    
    cards.forEach(card => {
        // Add magnetic effect
        card.addEventListener('mousemove', function(e) {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const deltaX = (x - centerX) / centerX;
            const deltaY = (y - centerY) / centerY;
            
            // Subtle tilt effect
            const tiltX = deltaY * 5;
            const tiltY = -deltaX * 5;
            
            card.style.transform = `
                translateY(-15px) 
                scale(1.02) 
                perspective(1000px) 
                rotateX(${tiltX}deg) 
                rotateY(${tiltY}deg)
            `;
        });
        
        card.addEventListener('mouseleave', function() {
            card.style.transform = 'translateY(0) scale(1) rotateX(0) rotateY(0)';
        });
        
        // Add ripple effect on click
        card.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: radial-gradient(circle, rgba(0, 255, 136, 0.3) 0%, transparent 70%);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple 0.6s ease-out;
                pointer-events: none;
            `;
            
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });
}

// Add enhanced ripple animation
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        0% {
            transform: scale(0);
            opacity: 1;
        }
        50% {
            transform: scale(2);
            opacity: 0.5;
        }
        100% {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    @keyframes glow-pulse {
        0%, 100% {
            text-shadow: 0 0 8px rgba(0, 255, 136, 0.3);
        }
        50% {
            text-shadow: 0 0 15px rgba(0, 255, 136, 0.6);
        }
    }
    
    @keyframes slide-in-right {
        from {
            transform: translateX(20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);

// ===================================
// SMOOTH SCROLLING
// ===================================
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            
            if (target) {
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset;
                const startPosition = window.pageYOffset;
                const distance = targetPosition - startPosition;
                const duration = 1000;
                let start = null;
                
                function animation(currentTime) {
                    if (start === null) start = currentTime;
                    const timeElapsed = currentTime - start;
                    const run = ease(timeElapsed, startPosition, distance, duration);
                    window.scrollTo(0, run);
                    if (timeElapsed < duration) requestAnimationFrame(animation);
                }
                
                function ease(t, b, c, d) {
                    t /= d / 2;
                    if (t < 1) return c / 2 * t * t + b;
                    t--;
                    return -c / 2 * (t * (t - 2) - 1) + b;
                }
                
                requestAnimationFrame(animation);
            }
        });
    });
}

// ===================================
// SECTION PARALLAX EFFECTS
// ===================================
function initSectionParallax() {
    const sections = document.querySelectorAll('.exercises-section, .features-section');
    
    if (sections.length === 0) {
        console.log('No sections found for parallax');
        return;
    }
    
    console.log('Initializing section parallax effects');
    
    let ticking = false;
    
    function updateSectionParallax() {
        const scrolled = window.pageYOffset;
        
        sections.forEach((section, index) => {
            const rect = section.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            
            // Only apply when section is in viewport
            if (rect.top < windowHeight && rect.bottom > 0) {
                // Different parallax speeds for variety
                const speeds = [0.1, 0.15, 0.08];
                const speed = speeds[index % speeds.length];
                
                const yPos = -(scrolled - rect.top) * speed;
                section.style.transform = `translate3d(0, ${yPos}px, 0)`;
                section.style.willChange = 'transform';
            }
        });
        
        ticking = false;
    }
    
    window.addEventListener('scroll', function() {
        if (!ticking) {
            window.requestAnimationFrame(updateSectionParallax);
            ticking = true;
        }
    });
}

// Section parallax is now initialized in the main DOMContentLoaded event

// ===================================
// UTILITY FUNCTIONS
// ===================================

// Debounce function for performance
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function for scroll events
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ===================================
// GRADIENT TEXT ANIMATION
// ===================================
function animateGradientText() {
    const gradientTexts = document.querySelectorAll('.gradient-text');
    
    gradientTexts.forEach(text => {
        let position = 0;
        setInterval(() => {
            position += 1;
            if (position > 200) position = 0;
            text.style.backgroundPosition = `${position}% 50%`;
        }, 30);
    });
}

// Initialize gradient animation
setTimeout(animateGradientText, 500);

// ===================================
// EXERCISE CARD SELECTION
// ===================================
window.selectExercise = function(exerciseType) {
    // Add exit animation
    const card = event.currentTarget;
    card.style.transform = 'scale(1.1)';
    card.style.opacity = '0.8';
    
    setTimeout(() => {
        window.location.href = `/exercise/${exerciseType}`;
    }, 200);
};

// ===================================
// PERFORMANCE OPTIMIZATION
// ===================================

// Reduce animations on low-performance devices
if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.querySelectorAll('*').forEach(el => {
        el.style.animation = 'none';
        el.style.transition = 'none';
    });
}

// Lazy load images
const lazyImages = document.querySelectorAll('img[data-src]');
const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.removeAttribute('data-src');
            imageObserver.unobserve(img);
        }
    });
});

lazyImages.forEach(img => imageObserver.observe(img));

// ===================================
// HAMBURGER MENU FUNCTIONALITY
// ===================================
function initNavigationDropdown() {
    const hamburgerMenu = document.getElementById('hamburger-menu');
    const userDropdown = document.getElementById('user-dropdown');
    const navUser = document.querySelector('.nav-user');
    
    console.log('Initializing hamburger menu...');
    console.log('hamburgerMenu:', hamburgerMenu);
    console.log('userDropdown:', userDropdown);
    
    if (!hamburgerMenu || !userDropdown) {
        console.log('Hamburger menu elements not found!');
        return;
    }
    
    let isMenuOpen = false;
    
    // Toggle hamburger menu on click
    hamburgerMenu.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log('Hamburger menu clicked! Current state:', isMenuOpen);
        
        isMenuOpen = !isMenuOpen;
        
        if (isMenuOpen) {
            console.log('Opening menu...');
            // Add show class and active class
            userDropdown.classList.add('show');
            hamburgerMenu.classList.add('active');
            if (navUser) navUser.classList.add('active');
        } else {
            console.log('Closing menu...');
            // Remove show class and active class
            userDropdown.classList.remove('show');
            hamburgerMenu.classList.remove('active');
            if (navUser) navUser.classList.remove('active');
        }
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (!hamburgerMenu.contains(e.target) && !userDropdown.contains(e.target) && isMenuOpen) {
            console.log('Clicking outside, closing menu...');
            isMenuOpen = false;
            userDropdown.classList.remove('show');
            hamburgerMenu.classList.remove('active');
            if (navUser) navUser.classList.remove('active');
        }
    });
    
    // Close dropdown on escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && isMenuOpen) {
            isMenuOpen = false;
            userDropdown.classList.remove('show');
            hamburgerMenu.classList.remove('active');
            if (navUser) navUser.classList.remove('active');
        }
    });
    
    // Handle dropdown item clicks with enhanced animations
    const dropdownItems = document.querySelectorAll('.dropdown-item');
    dropdownItems.forEach(item => {
        item.addEventListener('click', function(e) {
            // Add ripple effect
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: radial-gradient(circle, rgba(0, 255, 136, 0.3) 0%, transparent 70%);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple 0.6s ease-out;
                pointer-events: none;
            `;
            
            this.appendChild(ripple);
            
            // Add glow effect
            this.style.textShadow = '0 0 15px rgba(0, 255, 136, 0.5)';
            this.style.transform = 'translateX(8px) scale(1.02)';
            
            setTimeout(() => {
                ripple.remove();
                this.style.textShadow = '';
                this.style.transform = '';
            }, 600);
        });
        
        // Add hover sound effect (visual feedback)
        item.addEventListener('mouseenter', function() {
            this.style.transform = 'translateX(8px) scale(1.02)';
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.transform = '';
        });
    });
}

// ===================================
// FALLBACK HAMBURGER MENU FUNCTIONALITY
// ===================================
// Simple fallback in case the main function doesn't work
function initFallbackHamburgerMenu() {
    const hamburgerMenu = document.getElementById('hamburger-menu');
    const userDropdown = document.getElementById('user-dropdown');
    
    if (!hamburgerMenu || !userDropdown) return;
    
    // Simple click handler
    hamburgerMenu.onclick = function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        if (userDropdown.classList.contains('show')) {
            userDropdown.classList.remove('show');
            hamburgerMenu.classList.remove('active');
        } else {
            userDropdown.classList.add('show');
            hamburgerMenu.classList.add('active');
        }
    };
    
    // Close on outside click
    document.onclick = function(e) {
        if (!hamburgerMenu.contains(e.target) && !userDropdown.contains(e.target)) {
            userDropdown.classList.remove('show');
            hamburgerMenu.classList.remove('active');
        }
    };
}

// Initialize fallback after a delay
setTimeout(initFallbackHamburgerMenu, 2000);

// ===================================
// EXPORT FOR GLOBAL USE
// ===================================
window.FitnessApp = {
    debounce,
    throttle,
    selectExercise: window.selectExercise
};

// ===================================
// FLASH MESSAGES AUTO-DISMISS
// ===================================
function initFlashMessages() {
    const flashMessages = document.querySelectorAll('.flash-message');
    
    flashMessages.forEach(message => {
        // Add close button
        const closeButton = document.createElement('button');
        closeButton.innerHTML = '×';
        closeButton.style.cssText = `
            position: absolute;
            top: 5px;
            right: 10px;
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            opacity: 0.7;
            transition: opacity 0.2s;
        `;
        closeButton.addEventListener('mouseenter', () => closeButton.style.opacity = '1');
        closeButton.addEventListener('mouseleave', () => closeButton.style.opacity = '0.7');
        closeButton.addEventListener('click', () => dismissMessage(message));
        
        message.style.position = 'relative';
        message.appendChild(closeButton);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            dismissMessage(message);
        }, 5000);
    });
}

function dismissMessage(message) {
    message.style.animation = 'slideOut 0.3s ease-in-out forwards';
    setTimeout(() => {
        if (message.parentNode) {
            message.parentNode.removeChild(message);
        }
    }, 300);
}

// Add slideOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

console.log('🏋️ Motivational Fitness App Loaded');
console.log('💪 Train. Transform. Triumph.');