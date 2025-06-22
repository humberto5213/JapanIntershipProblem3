import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, transform, filters
import random

class DigitGenerator:
    def __init__(self):
        self.image_size = 28
        self.digit_templates = self._create_digit_templates()
    
    def _create_digit_templates(self):
        """Create basic templates for each digit 0-9"""
        templates = {}
        
        # Template for digit 0
        templates[0] = self._create_zero_template()
        templates[1] = self._create_one_template()
        templates[2] = self._create_two_template()
        templates[3] = self._create_three_template()
        templates[4] = self._create_four_template()
        templates[5] = self._create_five_template()
        templates[6] = self._create_six_template()
        templates[7] = self._create_seven_template()
        templates[8] = self._create_eight_template()
        templates[9] = self._create_nine_template()
        
        return templates
    
    def _create_zero_template(self):
        """Create template for digit 0"""
        img = np.zeros((28, 28))
        # Draw outer ellipse
        rr, cc = draw.ellipse(14, 14, 10, 6)
        img[rr, cc] = 1.0
        # Draw inner ellipse (hollow)
        rr, cc = draw.ellipse(14, 14, 7, 4)
        img[rr, cc] = 0.0
        return img
    
    def _create_one_template(self):
        """Create template for digit 1"""
        img = np.zeros((28, 28))
        # Draw vertical line
        rr, cc = draw.line(5, 14, 23, 14)
        img[rr, cc] = 1.0
        # Add slight angle at top
        rr, cc = draw.line(5, 14, 8, 12)
        img[rr, cc] = 1.0
        return img
    
    def _create_two_template(self):
        """Create template for digit 2"""
        img = np.zeros((28, 28))
        # Top curve
        rr, cc = draw.ellipse(8, 14, 4, 8)
        img[rr, cc] = 1.0
        rr, cc = draw.ellipse(8, 14, 2, 6)
        img[rr, cc] = 0.0
        # Diagonal line
        rr, cc = draw.line(12, 20, 20, 8)
        img[rr, cc] = 1.0
        # Bottom line
        rr, cc = draw.line(20, 8, 20, 20)
        img[rr, cc] = 1.0
        return img
    
    def _create_three_template(self):
        """Create template for digit 3"""
        img = np.zeros((28, 28))
        # Top curve
        rr, cc = draw.ellipse(8, 16, 4, 6)
        img[rr, cc] = 1.0
        rr, cc = draw.ellipse(8, 16, 2, 4)
        img[rr, cc] = 0.0
        # Middle line
        rr, cc = draw.line(14, 14, 14, 18)
        img[rr, cc] = 1.0
        # Bottom curve
        rr, cc = draw.ellipse(20, 16, 4, 6)
        img[rr, cc] = 1.0
        rr, cc = draw.ellipse(20, 16, 2, 4)
        img[rr, cc] = 0.0
        return img
    
    def _create_four_template(self):
        """Create template for digit 4"""
        img = np.zeros((28, 28))
        # Vertical line (right)
        rr, cc = draw.line(5, 18, 23, 18)
        img[rr, cc] = 1.0
        # Diagonal line
        rr, cc = draw.line(5, 8, 15, 18)
        img[rr, cc] = 1.0
        # Horizontal line
        rr, cc = draw.line(15, 8, 15, 22)
        img[rr, cc] = 1.0
        return img
    
    def _create_five_template(self):
        """Create template for digit 5"""
        img = np.zeros((28, 28))
        # Top horizontal line
        rr, cc = draw.line(5, 8, 5, 20)
        img[rr, cc] = 1.0
        # Vertical line (left part)
        rr, cc = draw.line(5, 8, 14, 8)
        img[rr, cc] = 1.0
        # Middle horizontal line
        rr, cc = draw.line(14, 8, 14, 18)
        img[rr, cc] = 1.0
        # Bottom curve
        rr, cc = draw.ellipse(20, 16, 4, 6)
        img[rr, cc] = 1.0
        rr, cc = draw.ellipse(20, 16, 2, 4)
        img[rr, cc] = 0.0
        return img
    
    def _create_six_template(self):
        """Create template for digit 6"""
        img = np.zeros((28, 28))
        # Outer ellipse
        rr, cc = draw.ellipse(16, 14, 8, 6)
        img[rr, cc] = 1.0
        # Inner ellipse (bottom part)
        rr, cc = draw.ellipse(18, 14, 4, 4)
        img[rr, cc] = 0.0
        # Top curve
        rr, cc = draw.ellipse(10, 12, 6, 4)
        img[rr, cc] = 1.0
        rr, cc = draw.ellipse(10, 12, 4, 2)
        img[rr, cc] = 0.0
        return img
    
    def _create_seven_template(self):
        """Create template for digit 7"""
        img = np.zeros((28, 28))
        # Top horizontal line
        rr, cc = draw.line(5, 8, 5, 20)
        img[rr, cc] = 1.0
        # Diagonal line
        rr, cc = draw.line(5, 20, 23, 12)
        img[rr, cc] = 1.0
        return img
    
    def _create_eight_template(self):
        """Create template for digit 8"""
        img = np.zeros((28, 28))
        # Top ellipse
        rr, cc = draw.ellipse(10, 14, 4, 6)
        img[rr, cc] = 1.0
        rr, cc = draw.ellipse(10, 14, 2, 4)
        img[rr, cc] = 0.0
        # Bottom ellipse
        rr, cc = draw.ellipse(18, 14, 5, 7)
        img[rr, cc] = 1.0
        rr, cc = draw.ellipse(18, 14, 3, 5)
        img[rr, cc] = 0.0
        return img
    
    def _create_nine_template(self):
        """Create template for digit 9"""
        img = np.zeros((28, 28))
        # Outer ellipse
        rr, cc = draw.ellipse(12, 14, 8, 6)
        img[rr, cc] = 1.0
        # Inner ellipse (top part)
        rr, cc = draw.ellipse(10, 14, 4, 4)
        img[rr, cc] = 0.0
        # Bottom curve
        rr, cc = draw.ellipse(18, 16, 6, 4)
        img[rr, cc] = 1.0
        rr, cc = draw.ellipse(18, 16, 4, 2)
        img[rr, cc] = 0.0
        return img
    
    def _add_variation(self, template):
        """Add random variations to make the digit look more handwritten"""
        img = template.copy()
        
        # Add random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        img = transform.rotate(img, angle, preserve_range=True)
        
        # Add random scaling
        scale = np.random.uniform(0.8, 1.2)
        img = transform.rescale(img, scale, preserve_range=True, anti_aliasing=True)
        
        # Ensure image is still 28x28
        if img.shape != (28, 28):
            img = transform.resize(img, (28, 28), preserve_range=True)
        
        # Add random translation
        shift_x = np.random.randint(-3, 4)
        shift_y = np.random.randint(-3, 4)
        img = np.roll(img, shift_x, axis=0)
        img = np.roll(img, shift_y, axis=1)
        
        # Add slight noise
        noise = np.random.normal(0, 0.1, img.shape)
        img = img + noise
        
        # Add some thickness variation
        if np.random.random() > 0.5:
            img = filters.gaussian(img, sigma=0.5)
        
        # Clip values to [0, 1]
        img = np.clip(img, 0, 1)
        
        return img
    
    def generate_digit_images(self, digit, count=5):
        """Generate multiple variations of a specific digit"""
        if digit not in range(10):
            raise ValueError("Digit must be between 0 and 9")
        
        template = self.digit_templates[digit]
        images = []
        
        for _ in range(count):
            varied_img = self._add_variation(template)
            images.append(varied_img)
        
        return images 