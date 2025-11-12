import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from deepface import DeepFace

# Load pretrained models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
model = YOLO('models/yolov8n.pt')

TARGET_CLASSES = {
    27: 'backpack',
    24: 'handbag',
    26: 'umbrella',
    25: 'suitcase',
    31: 'hat', # Not official, but some weights can include hat detection
    35: 'tie',
}

# Load precomputed models
with open('models/face_data.pkl', 'rb') as f:
    face_data = pickle.load(f)
dataset_embeddings = np.array([entry['embedding'] for entry in face_data])
dataset_labels = [entry['label'] for entry in face_data]

def load_embeddings(path='./models/face_data.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['models'], data['labels']


def extract_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        face = mtcnn(img)
        if face is None:
            return None
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = facenet(face).cpu().numpy()[0]
        return embedding
    except Exception as e:
        print(f"[!] Error: {e}")
        return None

def find_closest_match(embedding):
    if embedding is None:
        return "No face detected", 0.0
    sims = cosine_similarity([embedding], dataset_embeddings)
    best_idx = np.argmax(sims)
    similarity_score = float(sims[0][best_idx]) * 100
    return dataset_labels[best_idx], similarity_score

def detect_objects(image_path):
    results = model(image_path)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls_id]

        if cls_id in TARGET_CLASSES or name.lower() in ['backpack', 'handbag', 'tie']:
            detections.append({
                'label': name,
                'confidence': round(conf * 100, 1)
            })

    return detections

def deepface_detect(path):
    global top_object
    try:
        # Run DeepFace analysis
        analysis = DeepFace.analyze(
            img_path=path,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False
        )

        # Run object detection
        object_info = detect_objects(path)
        print("[✓] Object detection result:", object_info)

        top_object = object_info[0] if object_info else None

        gender_scores = analysis[0]['gender']
        likely_gender = max(gender_scores, key=gender_scores.get)

        extra_info = {
            'age': analysis[0].get('age', 'N/A'),
            'gender': 'Female' if likely_gender == 'Woman' else 'Male',
            'dominant_emotion': analysis[0].get('dominant_emotion', 'N/A')
        }
    except Exception as e:
        print(f"[!] DeepFace Error: {e}")
        extra_info = None

    return top_object, extra_info

