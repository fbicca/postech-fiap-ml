#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treinamento e Geração de Evidências – CNN Pneumonia (TensorFlow/Keras)
Autor: Edmilson Teixeira
Atualização: 2025-11-05

Este script unifica o pipeline:
1. Treinamento da CNN (com data augmentation e class_weight)
2. Geração de evidências e métricas (matriz, ROC, relatório, PDF, etc.)
"""

import os
import json
import datetime as dt
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

# ===========================================================
# 🔧 Funções auxiliares
# ===========================================================

def grid_samples(images, y_true, y_prob, class_names, k=12, cols=4, out_path="grid_preds.png"):
    """Salva um grid de k amostras com rótulo verdadeiro e predição."""
    import math
    if hasattr(images, "numpy"):
        images = images.numpy()
    if images.dtype != np.uint8:
        images = (images * 255).clip(0, 255).astype("uint8")

    k = min(k, images.shape[0])
    cols = max(1, cols)
    rows = math.ceil(k / cols)
    plt.figure(figsize=(cols*3, rows*3))
    for i in range(k):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(images[i])
        true_label = class_names[int(y_true[i])]
        prob_pos = float(y_prob[i])
        pred_idx = int(prob_pos >= 0.5)
        pred_label = class_names[pred_idx]
        ax.set_title(f"T:{true_label}\nP:{pred_label} ({prob_pos:.2f})", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


# ===========================================================
# 🚀 Função principal
# ===========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Treinar CNN e gerar evidências – Pneumonia.")
    parser.add_argument("--data_dir", type=str, default="data", help="Diretório com pastas train/val/test")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--outdir", type=str, default="evidencias_pneumonia")
    args = parser.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # =======================================================
    # 📂 Dataset e Augmentation
    # =======================================================
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_ds = datagen.flow_from_directory(
        os.path.join(args.data_dir, "train"),
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="binary"
    )

    val_ds = datagen.flow_from_directory(
        os.path.join(args.data_dir, "val"),
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="binary"
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(args.data_dir, "test"),
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False
    ).map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

    class_names = list(train_ds.class_indices.keys())
    print(f"Classes: {class_names}")

    # =======================================================
    # ⚖️ Pesos de classe automáticos
    # =======================================================
    labels = train_ds.classes
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(weights))
    print("Pesos de classe:", class_weights)

    # =======================================================
    # 🧠 Modelo CNN
    # =======================================================
    base_model = tf.keras.applications.MobileNetV2(input_shape=(args.img_size, args.img_size, 3),
                                                  include_top=False, weights="imagenet")
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # =======================================================
    # 🏋️‍♂️ Treinamento
    # =======================================================
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights
    )

    # =======================================================
    # 💾 Salva modelo
    # =======================================================
    model_dir = Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_{ts}.h5"
    model.save(model_path)
    print(f"✅ Modelo salvo em {model_path}")

    # =======================================================
    # 📊 Avaliação no conjunto de teste
    # =======================================================
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_prob = model.predict(test_ds).squeeze()
    y_pred = (y_prob >= args.threshold).astype(int)

    # Diagnóstico de distribuição
    unique, counts = np.unique(y_pred, return_counts=True)
    print("Distribuição de predições:", dict(zip(unique, counts)))

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusão – Pneumonia CNN")
    cm_path = outdir / f"matriz_confusao_{ts}.png"
    plt.savefig(cm_path, dpi=140); plt.close()

    # Curva ROC
    roc_path = outdir / f"roc_curve_{ts}.png"
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.legend(); plt.tight_layout()
    plt.savefig(roc_path, dpi=140); plt.close()

    # Classification report
    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    report_path = outdir / f"classification_report_{ts}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    # Grid sample
    sample_images, sample_labels = next(iter(test_ds.take(1)))
    grid_path = outdir / f"grid_preds_{ts}.png"
    grid_samples(sample_images, sample_labels.numpy(), y_prob[:sample_images.shape[0]], class_names, out_path=str(grid_path))

    # JSON resumo
    json_path = outdir / f"evidencias_pneumonia_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "model": str(model_path),
            "data_dir": args.data_dir,
            "threshold": args.threshold,
            "classes": class_names,
            "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc},
            "artifacts": {
                "cm_png": str(cm_path),
                "roc_png": str(roc_path),
                "grid_png": str(grid_path),
                "classification_report_txt": str(report_path)
            }
        }, f, indent=2, ensure_ascii=False)

    print("\n✅ Treinamento e evidências concluídos com sucesso!")
    print(f"- Modelo: {model_path}")
    print(f"- JSON:   {json_path}")
    print(f"- CM:     {cm_path}")
    print(f"- ROC:    {roc_path}")
    print(f"- GRID:   {grid_path}")
    print(f"- Report: {report_path}")

if __name__ == "__main__":
    main()
