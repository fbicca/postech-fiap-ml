# api-model-pneumonia.py
import os, io, json
from typing import List, Optional
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Forçar CPU e ocultar logs de erro do TensorFlow ---
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # força CPU (sem GPU)
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 3 = ERROR (oculta INFO/WARN)
# Opcional: para reprodutibilidade numérica, desative oneDNN:
# _os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
# Desabilita dispositivos GPU programaticamente (caso o ambiente exponha algo)
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "outputs" / "models"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

IMG_SIZE = (224, 224)

# -----------------------------
# Load model
# -----------------------------
model_path = None
for candidate in ["model.keras", "best_finetuned.keras", "best_feature_extractor.keras", "model.h5"]:
    p = MODEL_DIR / candidate
    if p.exists():
        model_path = p
        break

if model_path is None:
    raise RuntimeError("Nenhum modelo encontrado em outputs/models/")

model = tf.keras.models.load_model(model_path)

# Descobrir número de classes pelo output
num_classes = int(model.output_shape[-1])

# Carregar nomes de classes (se houver)
class_names: Optional[List[str]] = None
summary_json = REPORTS_DIR / "summary.json"
if summary_json.exists():
    try:
        meta = json.loads(summary_json.read_text(encoding="utf-8"))
        cls = meta.get("classes")
        if isinstance(cls, list) and len(cls) == num_classes:
            class_names = cls
    except Exception:
        pass

if class_names is None:
    class_names = [f"class_{i}" for i in range(num_classes)]

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="CNN Inference API",
    version="1.0.0",
    description="API para inferência de diagnóstico por imagem (CNN EfficientNet)."
)

# CORS (ajuste origins conforme o domínio do seu chatbot)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em prod, restrinja para seu domínio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    top_class: str
    top_prob: float
    probs: dict

# -----------------------------
# Utils
# -----------------------------
def load_img_bytes_to_tensor(file_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Falha ao abrir a imagem. Formatos aceitos: jpg, jpeg, png.")
    img = img.resize(IMG_SIZE)
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def softmax_to_dict(pred: np.ndarray, labels: List[str]) -> dict:
    pred = pred.reshape(-1)
    return {labels[i]: float(pred[i]) for i in range(len(labels))}

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(model_path.name),
        "num_classes": num_classes,
        "classes": class_names,
    }

@app.get("/classes")
def classes():
    return {"classes": class_names}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Envie uma imagem .jpg/.jpeg ou .png.")

    img_bytes = await file.read()
    x = load_img_bytes_to_tensor(img_bytes)
    probs = model.predict(x, verbose=0)[0]  # shape: (C,)

    # normalizar se necessário
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()

    probs_map = softmax_to_dict(probs, class_names)
    top_idx = int(np.argmax(probs))
    return PredictResponse(
        top_class=class_names[top_idx],
        top_prob=float(probs[top_idx]),
        probs=probs_map
    )
