from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
from digit_generator import DigitGenerator
import os

app = Flask(__name__)

# Initialize the digit generator
generator = DigitGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_digits():
    try:
        digit = int(request.json.get('digit', 0))
        if digit < 0 or digit > 9:
            return jsonify({'error': 'Digit must be between 0 and 9'}), 400
        
        # Generate 5 images of the specified digit
        images = generator.generate_digit_images(digit, count=5)
        
        # Convert images to base64 for web display
        image_data = []
        for img in images:
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            image_data.append(f"data:image/png;base64,{img_str}")
        
        return jsonify({
            'success': True,
            'digit': digit,
            'images': image_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 