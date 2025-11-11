from flask import Flask, render_template, request
import os
from src import *

# Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    match_label = None
    match_score = None
    extra_info = None
    top_object = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            try:
                # Save uploaded file
                path = os.path.join(UPLOAD_FOLDER, file.filename)
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                file.save(path)

                # Get FaceNet embedding & match
                embedding = extract_embedding(path)
                if embedding is not None:
                    match_label, match_score = find_closest_match(embedding)

                # --- Convert .jfif to .jpg if needed ---
                if path.lower().endswith('.jfif'):
                    from PIL import Image
                    img = Image.open(path)
                    path_jpg = path.rsplit('.', 1)[0] + '.jpg'
                    img.convert('RGB').save(path_jpg, 'JPEG')
                    os.remove(path)  # remove .jfif file
                    path = path_jpg

                top_object, extra_info = deepface_detect(path)

            except Exception as e:
                print(f"[!] Error: {e}")
        else:
            print("[!] No file uploaded")

    return render_template(
        'index.html',
        label=match_label,
        score= round(match_score,2) if match_score else None,  # percentage
        extra_info=extra_info,
        top_object= top_object
    )

if __name__ == '__main__':
    app.run(debug=True)
