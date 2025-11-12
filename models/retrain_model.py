import os
import json
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class RetrainManager:
    def __init__(self,
                 base_dataset_csv="data/pre_prompt_filter_dataset.csv",
                 feedback_csv="data/feedback_log.csv",
                 logs_jsonl="data/filter_logs.jsonl",
                 model_dir="models",
                 embed_model_name="all-MiniLM-L6-v2"):
        self.base_dataset_csv = base_dataset_csv
        self.feedback_csv = feedback_csv
        self.logs_jsonl = logs_jsonl
        self.model_dir = model_dir
        self.embed_model_name = embed_model_name
        os.makedirs(self.model_dir, exist_ok=True)

    def _load_base(self):
        if os.path.exists(self.base_dataset_csv):
            return pd.read_csv(self.base_dataset_csv)[["prompt", "label"]].dropna()
        return pd.DataFrame(columns=["prompt", "label"])

    def _load_feedback(self):
        if os.path.exists(self.feedback_csv):
            fb = pd.read_csv(self.feedback_csv)
            rows = []
            for _, r in fb.iterrows():
                preserved = int(r.get("preserved_meaning", 0))
                if preserved:
                    rows.append({"prompt": r.get("filtered_prompt", ""), "label": "safe"})
                else:
                    rows.append({"prompt": r.get("original_prompt", ""), "label": "sensitive"})
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["prompt", "label"])

    def _load_logs(self):
        if os.path.exists(self.logs_jsonl):
            items = []
            with open(self.logs_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        j = json.loads(line)
                        orig = j.get("original_prompt") or j.get("prompt") or ""
                        label = j.get("risk_label") or j.get("risk") or j.get("label") or ""
                        if orig and label:
                            items.append({"prompt": orig, "label": label})
                    except Exception:
                        continue
            return pd.DataFrame(items)
        return pd.DataFrame(columns=["prompt", "label"])

    def load_combined(self):
        parts = [self._load_base(), self._load_feedback(), self._load_logs()]
        combined = pd.concat(parts, ignore_index=True).drop_duplicates("prompt").dropna()
        return combined

    def initial_train_and_save(self):
        """
        Train initial embedding model (downloaded via sentence-transformers),
        create classifier from base dataset and save artifacts.
        """
        df = self._load_base()
        if df.empty:
            raise ValueError("Base dataset CSV not found or empty. Build dataset first.")

        embed = SentenceTransformer(self.embed_model_name)
        X = embed.encode(df["prompt"].tolist(), show_progress_bar=True)
        le = LabelEncoder()
        y = le.fit_transform(df["label"].tolist())

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X, y)
        embed.save(os.path.join(self.model_dir, "prompt_embedding_model"))

        joblib.dump(clf, os.path.join(self.model_dir, "prompt_filter_classifier.pkl"))
        joblib.dump(le, os.path.join(self.model_dir, "label_encoder.pkl"))
        print("Initial training complete. Models saved to", self.model_dir)

    def retrain(self):
        df = self.load_combined()
        if df.empty:
            raise ValueError("No data available to retrain.")
        embed = SentenceTransformer(os.path.join(self.model_dir, "prompt_embedding_model"))
        X = embed.encode(df["prompt"].tolist(), show_progress_bar=True)
        le = LabelEncoder()
        y = le.fit_transform(df["label"].tolist())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        print("Retrain results:")
        print(classification_report(y_test, preds, target_names=le.classes_))

        joblib.dump(clf, os.path.join(self.model_dir, "prompt_filter_classifier.pkl"))
        joblib.dump(le, os.path.join(self.model_dir, "label_encoder.pkl"))
        print("Retraining finished and saved to", self.model_dir)
        return True