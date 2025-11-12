import os
import threading

OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)
try:
    import openai
    if OPENAI_KEY:
        openai.api_key = OPENAI_KEY
except Exception:
    openai = None

_local_pipe = None
_local_lock = threading.Lock()

def _init_local_paraphraser(model_name="Vamsi/T5_Paraphrase_Paws"):
    global _local_pipe
    if _local_pipe is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            _local_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
        except Exception:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            _local_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return _local_pipe

def paraphrase(text: str, use_openai_if_available: bool = False):
    """
    Paraphrase text: prefer OpenAI if key set and requested; otherwise local model fallback.
    Returns paraphrased text (string). Errors fall back to returning original text.
    """
    if use_openai_if_available and openai:
        try:
            prompt = f"Paraphrase the following prompt, preserving meaning but removing any PII or sensitive detail. Return only the paraphrased prompt.\n\n{text}"
            resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=256, temperature=0.6)
            return resp.choices[0].text.strip()
        except Exception as e:
            print("OpenAI paraphrase failed:", e)

    try:
        with _local_lock:
            pipe = _init_local_paraphraser()
        out = pipe(text, max_length=256, num_return_sequences=1)
        if isinstance(out, list) and len(out) > 0:
            return out[0].get("generated_text", text).strip()
    except Exception as e:
        print("Local paraphrase failed:", e)
    return text