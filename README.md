# 🎯 FaceCard AI

### *Human-Centric Facial Matching, Appearance Detection & Attribute Analysis*

FaceCard AI is an intelligent Flask-based application that combines **Face Recognition**, **DeepFace attribute analysis**, and **YOLOv8 object detection** to describe a person the same way a human would.

This system not only identifies *who* a person looks like — it explains *what they look like* in natural human terms (clothing, hairstyle, accessories, expression, etc.).

---

## 🚀 Features

### ✅ 1. **Face Matching (FaceNet Embeddings)**

* Extracts 512-dimensional embeddings using FaceNet.
* Matches an uploaded face against stored embeddings.
* Returns **closest identity + confidence score**.
* Includes embedding storage, update, and versioning.

### ✅ 2. **DeepFace Appearance Analysis**

* Predicts:

  * **Age**
  * **Gender (converted to Male/Female automatically)**
  * **Dominant Emotion**
* Provides reliable detection even when faces are partially visible (`enforce_detection=False`).

### ✅ 3. **YOLOv8 Object & Clothing Detection**

Detects:

* Backpacks
* Jackets
* Ties
* Glasses
* Hats
* Shirts
* Shoes
* Watches
* And 80+ COCO dataset classes

The top detected item is converted into a **human-friendly description**, e.g.:

> *“Wearing a baggy jacket”*
> *“Carrying a backpack”*
> *“Wearing glasses”*

### ✅ 4. **Human-Style Appearance Narration**

The system produces an appearance summary similar to how a human describes someone:

> **“Male, around 25–30, afro hairstyle, wearing a baggy outfit, and looks confident.”**

(You can extend this with RAG or LLM prompts.)

### ✅ 5. **Modern UI With Live Image Preview**

* No page reload
* Smooth preview retention
* Dynamic result cards
* Reset button implemented without refreshing the page

---

## 🛠️ Tech Stack

| Layer                  | Technologies                               |
| ---------------------- | ------------------------------------------ |
| **Backend**            | Flask, Python                              |
| **Face Recognition**   | FaceNet (Keras / TensorFlow)               |
| **Attribute Analysis** | DeepFace                                   |
| **Object Detection**   | YOLOv8 (Ultralytics)                       |
| **Frontend**           | HTML, CSS, JS (Live Preview, Result Cards) |

---

## 📦 Installation

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/LAVAN-N/FaceCard-AI.git
cd FaceCard-AI
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate  # Windows
```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### Dependencies used:

```
flask
deepface
ultralytics
numpy
opencv-python
tensorflow
pillow
```

---

## 🧠 Running the Application

```bash
python app.py
```

App will be available at:

```
http://127.0.0.1:5000/
```

---

## 📁 Project Structure

```
FaceMatch-AI/
│── app.py
│── embeddings/
│   └── face_data.pkl
│── static/
│── templates/
│   └── index.html
│── uploads/
│── detect.py
│── requirements.txt
│── README.md
```

---

## 🔍 Object Detection Example Output

```json
[
  {
    "label": "backpack",
    "confidence": 27.9
  }
]
```

---

## 🧑‍🔬 Future Enhancements

* 🔥 **RAG-based appearance enhancement**
* 🧥 Clothing segmentation (better outfit descriptions)
* 🧬 Body landmarks + pose detection
* 🎤 Artist branding insights based on appearance
* 🌐 API mode (JSON response + mobile integration)

---

## 🤝 Contributing

Pull requests are welcome!

---

## 🛡️ License

MIT License

---

## ⭐ If you like this project

Please **star the repository** — it helps a lot!
