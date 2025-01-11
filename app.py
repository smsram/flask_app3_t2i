import os
import asyncio
from flask import Flask, request, jsonify, send_file
import requests
from io import BytesIO

# Load environment variables (if using .env file)
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Get API key from environment variable
API_KEY = os.getenv('HUGGINGFACE_API_KEY')

if not API_KEY:
    raise ValueError("API key is not set in environment variables")

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    image_data = generate_image_from_huggingface(prompt)
    image_stream = BytesIO(image_data)
    image_stream.seek(0)

    response = send_file(image_stream, mimetype='image/png')
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

def generate_image_from_huggingface(prompt):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=data)
    return response.content

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
