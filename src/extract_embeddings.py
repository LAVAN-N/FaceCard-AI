import os
import pickle
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


DATASET_DIR = '../../data/raw_images'
OUTPUT_PICKLE = 'models/face_data.pkl'

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def extract_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        face = mtcnn(img)
        if face is None:
            print(f"[!] No face detected in: {image_path}")
            return None
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = facenet(face).cpu().numpy()
        return embedding[0]
    except Exception as e:
        print(f"[!] Error processing {image_path}: {e}")
        return None

def build_dataset():
    face_data = []

    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue

        for filename in os.listdir(person_path):
            if not filename.lower().endswith(('png', 'jpg', 'jpeg')):
                continue

            image_path = os.path.join(person_path, filename)
            embedding = extract_embedding(image_path)
            if embedding is not None:
                face_data.append({
                    'label': person,
                    'embedding': embedding
                })

    os.makedirs(os.path.dirname(OUTPUT_PICKLE), exist_ok=True)
    with open(OUTPUT_PICKLE, 'wb') as f:
        pickle.dump(face_data, f)

    print(f"[✓] Saved {len(face_data)} face models to {OUTPUT_PICKLE}")

if __name__ == '__main__':
    # Step 1: Build dataset models
    build_dataset()
