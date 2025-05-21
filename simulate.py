import os
import numpy as np
import onnxruntime as ort
from PIL import Image

# === Config ===
onnx_model_path = "onnx_export/clip_image_encoder.onnx"
positive_vec_path = "onnx_export/positive.npy"
negative_vec_path = "onnx_export/negative.npy"
image_folder = "samples"  # or any folder with test images

# Load ONNX model and text embeddings
session = ort.InferenceSession(onnx_model_path)
positive_vec = np.load(positive_vec_path)
negative_vec = np.load(negative_vec_path)

def preprocess_image(path):
    image = Image.open(path).convert("RGB").resize((224, 224))
    image_np = np.array(image).astype(np.float32) / 255.0

    # Normalize using CLIP's mean/std
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    image_np = (image_np - mean) / std

    # Convert to NCHW
    image_np = image_np.transpose(2, 0, 1)[None, :, :, :]  # [1, 3, 224, 224]
    return image_np.astype(np.float32)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def score_image(path):
    inputs = {"pixel_values": preprocess_image(path)}
    outputs = session.run(["image_embeds"], inputs)
    image_vec = outputs[0][0]  # [1, 512] -> [512]

    pos_sim = cosine(image_vec, positive_vec)
    neg_sim = cosine(image_vec, negative_vec)
    return (pos_sim - neg_sim) * 1000

# Score images
print("Scoring images using ONNX Runtime...")
for fname in os.listdir(image_folder):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    path = os.path.join(image_folder, fname)
    try:
        score = score_image(path)
        print(f"{fname}: {score:.2f}")
    except Exception as e:
        print(f"{fname}: ERROR - {e}")
