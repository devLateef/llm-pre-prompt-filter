import os
import json
import threading
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.filter_model import FilterModel
from models.paraphraser import paraphrase
from models.utils import mask_pii, replace_entities_with_fakes
from models.retrain_model import RetrainManager

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

FILTER_LOGS = os.path.join(DATA_DIR, "filter_logs.jsonl")
FEEDBACK_CSV = os.path.join(DATA_DIR, "feedback_log.csv")
BASE_DATA_CSV = os.path.join(DATA_DIR, "pre_prompt_filter_dataset.csv")

MODEL_DIR = "models"
THRESHOLD_ALLOW = 0.70
THRESHOLD_SANITIZE = 0.60

retrain_manager = RetrainManager(base_dataset_csv=BASE_DATA_CSV, model_dir=MODEL_DIR)
try:
    model = FilterModel(model_dir=MODEL_DIR)
except Exception as e:
    print("Warning: FilterModel not available yet. Run initial training via retrain_manager.initial_train_and_save().")
    model = None

app = FastAPI(title="Pre-Prompt Filter API")

class FilterRequest(BaseModel):
    prompt: str
    request_paraphrase_if_blocked: bool = True
    user_id: str | None = None

class FeedbackRequest(BaseModel):
    original_prompt: str
    filtered_prompt: str
    preserved_meaning: bool
    user_id: str | None = None
    comments: str | None = None

retrain_lock = threading.Lock()
retrain_in_progress = False

def log_filter(entry: dict):
    with open(FILTER_LOGS, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def log_feedback_csv(row: dict):
    header = ["timestamp", "user_id", "original_prompt", "filtered_prompt", "preserved_meaning", "comments"]
    write_header = not os.path.exists(FEEDBACK_CSV)
    with open(FEEDBACK_CSV, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        line = ",".join([str(row.get(h, "")) for h in header])
        f.write(line + "\n")

@app.get("/")
def root():
    return {
        "message": "Pre-Prompt Filter API is running.",
        "available_endpoints": ["/filter", "/feedback", "/retrain"],
        "docs": "Visit /docs for the interactive Swagger UI."
    }

@app.post("/filter")
def filter_endpoint(req: FilterRequest):
    global model
    text = req.prompt.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty prompt")

    if model is None:
        masked, entities = mask_pii(text)
        replaced = replace_entities_with_fakes(masked, entities)
        return {"original_prompt": text, "risk_label": "unknown", "confidence": 0.0, "recommended_action": "sanitize", "filtered_prompt": replaced}

    label, confidence, probs = model.predict(text)
    action = "block"
    filtered = "[BLOCKED_PROMPT]"

    if label == "safe" and confidence >= THRESHOLD_ALLOW:
        action = "allow"
        filtered = text
    elif label == "sensitive" and confidence >= THRESHOLD_SANITIZE:
        action = "sanitize"
        masked, entities = mask_pii(text)
        filtered = replace_entities_with_fakes(masked, entities)
    else:
        # blocked or low confidence
        action = "block"
        filtered = "[BLOCKED_PROMPT]"
        if req.request_paraphrase_if_blocked:
            try:
                masked, entities = mask_pii(text)
                replaced = replace_entities_with_fakes(masked, entities)
                paraphrased = paraphrase(replaced, use_openai_if_available=False)
                if paraphrased and paraphrased.strip():
                    filtered = paraphrased
                    action = "sanitized_paraphrase"
            except Exception as e:
                print("Paraphrase failed:", e)

    log_entry = {
        "timestamp": time.time(),
        "user_id": req.user_id or "",
        "original_prompt": text,
        "filtered_prompt": filtered,
        "risk_label": label,
        "confidence": confidence,
        "action": action
    }
    log_filter(log_entry)

    return {
        "original_prompt": text,
        "risk_label": label,
        "confidence": round(confidence, 3),
        "recommended_action": action,
        "filtered_prompt": filtered
    }

@app.post("/feedback")
def feedback_endpoint(fb: FeedbackRequest):
    row = {
        "timestamp": time.time(),
        "user_id": fb.user_id or "",
        "original_prompt": fb.original_prompt,
        "filtered_prompt": fb.filtered_prompt,
        "preserved_meaning": int(bool(fb.preserved_meaning)),
        "comments": fb.comments or ""
    }
    log_feedback_csv(row)
    return {"status": "ok", "message": "Feedback recorded."}

@app.post("/retrain")
def retrain_endpoint(background: bool = True):
    global retrain_in_progress, model
    if retrain_in_progress:
        return {"status": "busy", "message": "Retraining already in progress."}

    def _run():
        global retrain_in_progress, model
        try:
            retrain_in_progress = True
            try:
                retrain_manager.initial_train_and_save()
            except Exception as e:
                print("initial_train_and_save:", e)
            retrain_manager.retrain()
            model = FilterModel(model_dir=MODEL_DIR)
        except Exception as e:
            print("Retrain error:", e)
        finally:
            retrain_in_progress = False

    if background:
        threading.Thread(target=_run, daemon=True).start()
        return {"status": "started", "message": "Retraining started in background."}
    else:
        _run()
        return {"status": "completed", "message": "Retraining finished."}

@app.get("/retrain/status")
def retrain_status():
    return {"in_progress": retrain_in_progress}