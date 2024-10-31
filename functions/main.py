from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import logging.config

import settings
from model import generate_captions, load_model
from tagging import tag_captions

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Configure logging
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Load the pre-trained BLIP model and processor from Hugging Face
processor, model = load_model(settings.settings.blip_model_name)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Route for health check
@app.route('/', methods=['GET'])
def home():
    return "Image Captioning API is running!"

# Route to caption an image
@app.route('/caption', methods=['POST'])
def caption_image():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
     # Get user suggestion from the request (optional)
    suggestion = request.form.get('suggestion', '').strip()

    try:
        # Open the image file using PIL
        image = Image.open(io.BytesIO(file.read()))

        # If there's a user suggestion, include it in the input
        prompt_text = suggestion if suggestion else ""

        # Preprocess the image and generate a caption
        captions = generate_captions(model, processor, image, device, prompt_text)
        
        # Process each caption with spaCy and add emojis
        captions = tag_captions(captions)
        
        # Return the generated captions as a JSON response
        return jsonify({'captions': captions})


    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
