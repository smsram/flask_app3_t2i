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
async def generate_image():
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Asynchronously call the Hugging Face API (non-blocking)
    image_data = await asyncio.to_thread(generate_image_from_huggingface, prompt)

    # Use BytesIO to convert binary data for sending as a file
    image_stream = BytesIO(image_data)
    image_stream.seek(0)

    return send_file(image_stream, mimetype='image/png')

def generate_image_from_huggingface(prompt):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=data)
    return response.content

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
