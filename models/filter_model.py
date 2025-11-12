import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

class FilterModel:
    """
    Wrapper around SentenceTransformer embeddings + classifier.
    Expects artifacts in model_dir:
      - prompt_embedding_model/   (SentenceTransformer dir)
      - prompt_filter_classifier.pkl
      - label_encoder.pkl
    """

    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.embed_path = os.path.join(model_dir, "prompt_embedding_model")
        self.clf_path = os.path.join(model_dir, "prompt_filter_classifier.pkl")
        self.encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        self._load()

    def _load(self):
        if not os.path.isdir(self.embed_path):
            raise FileNotFoundError(f"Embedding model directory not found: {self.embed_path}")
        if not os.path.exists(self.clf_path) or not os.path.exists(self.encoder_path):
            raise FileNotFoundError("Classifier or encoder not found. Please train initial models first.")
        self.embed = SentenceTransformer(self.embed_path)
        self.clf = joblib.load(self.clf_path)
        self.encoder = joblib.load(self.encoder_path)

    def predict(self, prompt: str):
        emb = self.embed.encode([prompt])
        probs = self.clf.predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        label = self.encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])
        return label, confidence, probs

    def reload(self):
        self.clf = joblib.load(self.clf_path)
        self.encoder = joblib.load(self.encoder_path)