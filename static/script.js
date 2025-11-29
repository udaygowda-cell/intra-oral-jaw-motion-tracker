// File input handling
document.getElementById('file-input').addEventListener('change', function(e) {
    const fileName = e.target.files[0] ? e.target.files[0].name : 'No file chosen';
    document.getElementById('file-name').textContent = fileName;
});

// Form submission with loading state
document.getElementById('upload-form').addEventListener('submit', function(e) {
    const submitButton = this.querySelector('.btn-analyze');
    const fileInput = document.getElementById('file-input');
    
    if (!fileInput.files.length) {
        e.preventDefault();
        alert('Please select a file to upload.');
        return;
    }
    
    // Show loading state
    submitButton.classList.add('loading');
    
    // Simulate processing delay for better UX
    setTimeout(() => {
        // Form will submit normally
    }, 500);
});

// Jaw animation enhancement
document.addEventListener('DOMContentLoaded', function() {
    const lowerJaw = document.querySelector('.lower-jaw');
    
    // Add subtle animation variations
    function animateJaw() {
        const delay = Math.random() * 2000 + 1000; // 1-3 seconds
        const height = 30 + Math.random() * 20; // 30-50px movement
        
        lowerJaw.style.transition = `transform ${delay}ms ease-in-out`;
        lowerJaw.style.transform = `translateY(-${height}px)`;
        
        setTimeout(() => {
            lowerJaw.style.transform = 'translateY(0)';
            setTimeout(animateJaw, delay);
        }, delay);
    }
    
    // Start animation
    setTimeout(animateJaw, 1000);
});

// Smooth scrolling for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            window.scrollTo({
                top: target.offsetTop - 80,
                behavior: 'smooth'
            });
        }
    });
});

// Feature card animations
const featureCards = document.querySelectorAll('.feature-card');
featureCards.forEach((card, index) => {
    // Add delay for staggered animation
    card.style.animationDelay = `${index * 0.2}s`;
    
    // Add hover effect enhancement
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-10px) scale(1.02)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0) scale(1)';
    });
});

// Activity card animations
const activityCards = document.querySelectorAll('.activity-card');
activityCards.forEach((card, index) => {
    card.style.animationDelay = `${index * 0.2}s`;
});