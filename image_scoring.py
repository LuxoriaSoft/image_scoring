import os
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Setup model & processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create prompts
positive_prompts = ["an outstanding picture"]
negative_prompts = ["a horrible picture"]

def get_prompt_embedding(prompts):
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        return model.get_text_features(**inputs).cpu().numpy().mean(axis=0)

# Compute embeddings once
positive_vec = get_prompt_embedding(positive_prompts)
negative_vec = get_prompt_embedding(negative_prompts)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def score_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_feat = model.get_image_features(**inputs).cpu().numpy().flatten()
    pos_sim = cosine_similarity(image_feat, positive_vec)
    neg_sim = cosine_similarity(image_feat, negative_vec)
    return (pos_sim - neg_sim) * 1000

# Load assets
image_folder = "samples"  # Samples to be scored
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

results = [{"image": os.path.basename(p), "score": score_image(p)} for p in image_paths]

# Save results
df = pd.DataFrame(results).sort_values(by="score", ascending=False)
df.to_csv("scores.csv", index=False)
print("Scores saved !")
