from flask import Flask, render_template, request, jsonify
import os
from utils.functionalities import extract_embedding, find_closest_match, deepface_detect
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('image')
    if not file or file.filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Save file
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # Get match info
        result = perform_match(path)
        return render_template('index.html', **result)

    except Exception as e:
        print(f"[!] Error: {e}")
        return jsonify({"error": str(e)}), 500


def perform_match(path):
    """Handles embedding extraction, comparison, and detection."""
    embedding = extract_embedding(path)
    match_label, match_score = (None, None)
    if embedding is not None:
        match_label, match_score = find_closest_match(embedding)

    # Convert .jfif → .jpg if needed
    if path.lower().endswith('.jfif'):
        img = Image.open(path)
        path_jpg = path.rsplit('.', 1)[0] + '.jpg'
        img.convert('RGB').save(path_jpg, 'JPEG')
        os.remove(path)
        path = path_jpg

    # Run DeepFace or your detection
    top_object, extra_info = deepface_detect(path)

    return {
        "label": match_label,
        "score": round(match_score, 2) if match_score else None,
        "extra_info": extra_info,
        "top_object": top_object
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
