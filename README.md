# Handwritten Digit Generator Web App

A beautiful web application that generates realistic handwritten digits (0-9) similar to the famous MNIST dataset. Users can select any digit and the app will generate 5 unique variations with realistic handwriting characteristics.

## Features

- 🎨 Generate handwritten digits 0-9
- 🖼️ Creates 5 unique variations per digit
- 📱 Responsive, modern web interface
- 🎯 MNIST-compatible format (28×28 grayscale images)
- ⚡ Real-time generation with visual feedback
- 🎲 Random variations including rotation, scaling, translation, and noise

## How It Works

The application uses a custom digit generator that:
1. Creates base templates for each digit (0-9)
2. Applies random transformations to simulate natural handwriting variations:
   - Random rotation (-15° to +15°)
   - Random scaling (80% to 120%)
   - Random translation (small shifts)
   - Gaussian noise for texture
   - Optional blur for thickness variation

## Screenshots

The web interface features:
- Modern gradient design with smooth animations
- Interactive digit selection buttons
- Loading animations during generation
- Grid layout for displaying generated images
- Mobile-responsive design

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser and visit:**
   ```
   http://localhost:5000
   ```

## Project Structure

```
JapanIntershipProjectProblem3/
├── app.py                 # Main Flask application
├── digit_generator.py     # Digit generation logic
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface template
└── README.md             # This file
```

## Usage

1. **Select a Digit:** Click on any digit button (0-9)
2. **Generate Images:** Click the "Generate 5 Images" button
3. **View Results:** The app will display 5 unique variations of your selected digit
4. **Repeat:** Select different digits to generate more images

## Technical Details

### Dependencies
- **Flask:** Web framework for the backend API
- **NumPy:** Numerical computations and array operations
- **Pillow:** Image processing and format conversion
- **Matplotlib:** Additional image processing utilities
- **scikit-image:** Advanced image processing (drawing, transforms, filters)

### API Endpoints
- `GET /` - Serves the main web interface
- `POST /generate` - Generates digit images
  - Request: `{"digit": 0-9}`
  - Response: `{"success": true, "digit": X, "images": ["base64_image1", ...]}`

### Image Generation Process
1. **Template Creation:** Each digit has a unique template created using geometric shapes
2. **Variation Application:** Random transformations are applied to create natural variations
3. **Format Conversion:** Images are converted to base64 for web display
4. **MNIST Compatibility:** All images are 28×28 grayscale, matching MNIST format

## Customization

You can easily modify the digit generation by editing `digit_generator.py`:
- Adjust variation parameters (rotation angle, scaling range, etc.)
- Modify digit templates for different handwriting styles
- Change image size (currently 28×28 for MNIST compatibility)
- Add new transformation types

## Browser Compatibility

The web app works on all modern browsers including:
- Chrome/Chromium
- Firefox
- Safari
- Edge

## Performance

- **Generation Time:** ~1-2 seconds for 5 images
- **Image Size:** 28×28 pixels (MNIST standard)
- **Memory Usage:** Minimal - images are generated on-demand
- **Scalability:** Can easily handle multiple concurrent users

## Future Enhancements

Potential improvements could include:
- Save generated images to local storage
- Batch generation of multiple digits
- Export functionality (ZIP download)
- Advanced styling options
- Integration with actual ML models for more realistic generation
- Animation of the generation process

## License

This project is open source and available under the MIT License. 