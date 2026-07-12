# ============================================================
# UNIFIED PIPELINE — UI-agnostic core
# Binary (same as run_streamlit_v3) + summary (Qwen3-VL + Q-Former LoRA)
# + multi-class (EnhancedTemporalAdapterModel using generated summary)
# ============================================================
import os
import re
import random
import joblib
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import pipeline_base as pb
import binary_ensemble

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

RANDOM_SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

# Paths are overridable via environment variables so the repo stays machine-agnostic.
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
MULTICLASS_MODEL_DIR = os.getenv(
    "MULTICLASS_MODEL_DIR",
    "/content/drive/MyDrive/models_ucf_v2/hpo_best_models"
)
SUMMARIZATION_CHECKPOINT = os.getenv(
    "SUMMARIZATION_CHECKPOINT",
    "/content/drive/MyDrive/Project_VLM/ucf_qwen_v9_qformer/checkpoint-400"
)
BINARY_MODEL_DIR = os.getenv(
    "BINARY_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
)

FRAMES = int(os.getenv("FRAMES", "8"))

# Wire globals expected by pipeline_base functions
pb.DEVICE = DEVICE
pb.FRAMES = FRAMES

# ---------------- MODEL CACHE ----------------
_MODELS = None


def _log(msg, progress=None):
    if progress is not None:
        progress(msg)
    print(msg)


def load_models(progress=None):
    """Load all models once and cache them."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    # 1. Qwen3-VL backbone (shared for embedding + binary)
    _log("Loading Qwen3-VL embedding backbone...", progress)
    qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    ).to(DEVICE)
    qwen_model.eval()

    # 2. Summarization model (Qwen3-VL + Q-Former LoRA)
    _log("Loading summarization model...", progress)
    summary_model = None
    if os.path.exists(SUMMARIZATION_CHECKPOINT):
        try:
            summary_model = Qwen3VLForConditionalGeneration.from_pretrained(
                SUMMARIZATION_CHECKPOINT, torch_dtype=DTYPE, trust_remote_code=True
            ).to(DEVICE)
            summary_model.eval()
        except Exception as e:
            _log(f"Summarization model failed to load: {e}", progress)
    else:
        _log(f"Summarization checkpoint not found: {SUMMARIZATION_CHECKPOINT}", progress)

    # 3. Multi-class model (uses generated summary + frame embeddings)
    _log("Loading multi-class model...", progress)
    multiclass_checkpoint = os.path.join(
        MULTICLASS_MODEL_DIR, "multiclass_temporal_adapter_qwen3_instruct_improved.pt"
    )
    multiclass_encoders_path = os.path.join(MULTICLASS_MODEL_DIR, "multiclass_label_encoders.pkl")
    multiclass_model = None
    multiclass_encoders = None

    if os.path.exists(multiclass_checkpoint) and os.path.exists(multiclass_encoders_path):
        try:
            checkpoint = torch.load(multiclass_checkpoint, map_location="cpu", weights_only=True)
            config = checkpoint["config"]
            multiclass_encoders = joblib.load(multiclass_encoders_path)

            multiclass_model = pb.EnhancedTemporalAdapterModel(
                in_dim=config["in_dim"],
                proj_dim=config["proj_dim"],
                nhead=config["nhead"],
                adapter_dim=config["adapter_dim"],
                tcn_layers=config.get("tcn_layers", 3),
                attn_layers=config.get("attn_layers", 6),
                dropout=config["dropout"],
                n_ml=config["n_ml"],
                n_fine=config["n_fine"],
                n_sup=config["n_sup"],
                n_wpn=config["n_wpn"],
                n_loc=config["n_loc"],
                n_ppl=config["n_ppl"],
                n_queries=config.get("n_queries", 4),
                max_len=config.get("max_len", 64),
            ).to(DEVICE)
            multiclass_model.load_state_dict(checkpoint["model_state_dict"])
            multiclass_model.eval()
        except Exception as e:
            _log(f"Multi-class model failed to load: {e}", progress)
    else:
        _log(f"Multi-class checkpoint not found: {multiclass_checkpoint}", progress)

    # 4. Binary ensemble (same as run_streamlit_v3.py)
    _log("Loading binary ensemble...", progress)
    binary_models = None
    if os.path.exists(BINARY_MODEL_DIR):
        try:
            binary_models = binary_ensemble.load_binary_models(DEVICE, BINARY_MODEL_DIR)
        except Exception as e:
            _log(f"Binary ensemble failed to load: {e}", progress)
    else:
        _log(f"Binary model directory not found: {BINARY_MODEL_DIR}", progress)

    # Wire globals for pipeline_base
    pb.qwen_processor = qwen_processor
    pb.qwen_model = qwen_model
    pb.summary_model = summary_model

    _MODELS = {
        "qwen_processor": qwen_processor,
        "qwen_model": qwen_model,
        "summary_model": summary_model,
        "multiclass_model": multiclass_model,
        "multiclass_encoders": multiclass_encoders,
        "binary_models": binary_models,
    }
    return _MODELS


def _decode_multiclass(outputs, encoders):
    """Convert multi-class logits to human-readable labels."""
    result = {}
    # Multi-label actions
    if "action_ml" in outputs and encoders.get("mlb_action"):
        mlb = encoders["mlb_action"]
        proba = torch.sigmoid(outputs["action_ml"]).cpu().numpy()
        pred = (proba >= 0.5).astype(int)
        for i in range(len(pred)):
            if pred[i].sum() == 0:
                pred[i, proba[i].argmax()] = 1
        result["actions"] = list(mlb.inverse_transform(pred[0:1])[0])
    else:
        result["actions"] = []

    # Fine-grained action
    if "action_fine" in outputs and encoders.get("le_action_fine"):
        le = encoders["le_action_fine"]
        idx = outputs["action_fine"].argmax(1).cpu().numpy()
        result["activity"] = le.inverse_transform(idx)[0]
    else:
        result["activity"] = "Unknown"

    # Super-class action
    if "action_sup" in outputs and encoders.get("le_action_sup"):
        le = encoders["le_action_sup"]
        idx = outputs["action_sup"].argmax(1).cpu().numpy()
        result["category"] = le.inverse_transform(idx)[0]
    else:
        result["category"] = "Unknown"

    # Weapon
    if "weapon" in outputs and encoders.get("le_weapon"):
        le = encoders["le_weapon"]
        idx = outputs["weapon"].argmax(1).cpu().numpy()
        result["weapon"] = le.inverse_transform(idx)[0]
    else:
        result["weapon"] = "Unknown"

    # Location
    if "location" in outputs and encoders.get("le_location"):
        le = encoders["le_location"]
        idx = outputs["location"].argmax(1).cpu().numpy()
        result["location"] = le.inverse_transform(idx)[0]
    else:
        result["location"] = "Unknown"

    # People
    if "people" in outputs and encoders.get("le_people"):
        le = encoders["le_people"]
        idx = outputs["people"].argmax(1).cpu().numpy()
        result["people"] = le.inverse_transform(idx)[0]
    else:
        result["people"] = "Unknown"

    return result


def _clean_summary(text):
    """Strip model role markers and extra whitespace."""
    text = text.strip()
    if text.lower().startswith("assistant"):
        text = text[9:].strip()
    text = re.sub(r"\s+", " ", text)
    if not text.endswith("."):
        text += "."
    return text


def analyze_video(video_path, models=None):
    """Run the full pipeline on a single video and return results."""
    if models is None:
        models = load_models()

    qwen_model = models["qwen_model"]
    qwen_processor = models["qwen_processor"]
    summary_model = models["summary_model"]
    multiclass_model = models["multiclass_model"]
    multiclass_encoders = models["multiclass_encoders"]
    binary_models = models["binary_models"]

    # 1. Sample frames
    frames_bgr = pb.sample_frames(video_path)
    if not frames_bgr:
        return None

    # PIL images for binary embedding
    frames_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]

    # 2. Generate summary
    summary_text = "No summary available"
    if summary_model is not None:
        summary_text, _ = pb.generate_summary(frames_bgr)
    summary_text = _clean_summary(summary_text)

    # 3. Frame + text embeddings for multi-class
    frame_emb = pb.embed_frames(frames_bgr)
    text_emb = pb.embed_text(summary_text) if summary_text else torch.zeros(frame_emb.shape[-1])

    # 4. Multi-class prediction
    attrs = {
        "activity": "Unknown",
        "category": "Unknown",
        "weapon": "Unknown",
        "location": "Unknown",
        "people": "Unknown",
        "actions": [],
    }
    if multiclass_model is not None and multiclass_encoders is not None:
        mc_dtype = next(multiclass_model.parameters()).dtype
        frames_batch = frame_emb.unsqueeze(0).to(DEVICE, dtype=mc_dtype)
        text_batch = text_emb.unsqueeze(0).to(DEVICE, dtype=mc_dtype)
        with torch.no_grad():
            outputs = multiclass_model(frames_batch, text_batch)
        attrs = _decode_multiclass(outputs, multiclass_encoders)

    # 5. Binary classification
    binary_class = "UNKNOWN"
    binary_conf = 0.0
    binary_parts = {"simple": 0.0, "lora": 0.0}
    if binary_models is not None and qwen_model is not None and qwen_processor is not None:
        binary_class, binary_conf, binary_parts = binary_ensemble.predict_binary(
            frames_pil, binary_models, qwen_model, qwen_processor, DEVICE
        )

    return {
        "video_path": video_path,
        "frames": frames_pil,
        "binary_class": binary_class,
        "binary_confidence": float(binary_conf),
        "binary_parts": binary_parts,
        "summary": summary_text,
        "activity": attrs["activity"],
        "category": attrs["category"],
        "weapon": attrs["weapon"],
        "location": attrs["location"],
        "people": attrs["people"],
        "actions": attrs["actions"],
    }


def answer_question(ctx, question):
    """Simple rule-based VQA on the pipeline context."""
    q = question.lower()
    if any(w in q for w in ["normal", "anomalous", "status", "safe"]):
        return f"Status: {ctx['binary_class']} (confidence: {ctx['binary_confidence']:.2%})"
    if any(w in q for w in ["people", "person", "how many", "number", "count"]):
        return f"People: {ctx['people']}"
    if any(w in q for w in ["weapon", "gun", "knife", "armed", "used"]):
        return f"Weapon: {ctx['weapon']}"
    if any(w in q for w in ["location", "where", "place", "located"]):
        return f"Location: {ctx['location']}"
    if any(w in q for w in ["activity", "category", "type", "event", "what is happening", "what"]):
        return f"Activity: {ctx['activity']}. Category: {ctx['category']}. Summary: {ctx['summary']}"
    if any(w in q for w in ["action", "actions"]):
        return f"Actions: {', '.join(ctx['actions']) or 'None detected'}"
    return ctx["summary"]
