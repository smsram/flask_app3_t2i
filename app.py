import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from io import BytesIO
import requests
from dotenv import load_dotenv

# Load environment variables (if using .env file)
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from different origins

# Get API key from environment variable
API_KEY = os.getenv('HUGGINGFACE_API_KEY')

if not API_KEY:
    raise ValueError("API key is not set in environment variables")

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        # Parse the JSON data from the request
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Generate the image using the Hugging Face API
        image_data = generate_image_from_huggingface(prompt)

        if not image_data:
            return jsonify({'error': 'Failed to generate image from the Hugging Face model'}), 500

        # Prepare the image data for response
        image_stream = BytesIO(image_data)
        image_stream.seek(0)

        # Send the image as a response
        response = send_file(image_stream, mimetype='image/png')
        response.cache_control.no_cache = True
        response.cache_control.no_store = True
        response.cache_control.must_revalidate = True
        return response

    except Exception as e:
        print("Error while generating image:", e)
        return jsonify({'error': str(e)}), 500

def generate_image_from_huggingface(prompt):
    try:
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        data = {"inputs": prompt}

        # Call the Hugging Face API
        response = requests.post(API_URL, headers=headers, json=data)

        # Check for successful response
        if response.status_code != 200:
            print(f"Hugging Face API error: {response.status_code}, {response.text}")
            return None

        return response.content

    except requests.exceptions.RequestException as e:
        print("Request to Hugging Face API failed:", e)
        return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
