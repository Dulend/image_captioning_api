from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io
import base64
import os

os.environ['TRANSFORMERS_CACHE'] = './model_cache'

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-small", cache_dir="./model_cache")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-small", cache_dir="./model_cache")

app = Flask(__name__)

@app.route('/')
def home():
    return 'BLIP Captioning API is running!'

@app.route('/caption', methods=['POST'])
def caption_image():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
