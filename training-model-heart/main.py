#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerar evidências do modelo de risco cardíaco.
- Pré-processamento: nulos, One-Hot (get_dummies), split 70/30 (stratify), StandardScaler
- Treinamento: LogisticRegression(solver='liblinear')
- Avaliação: Accuracy, Precision, Recall, F1, AUC
- Saídas: JSON de métricas, classification_report, matriz de confusão (PNG), curva ROC (PNG),
          log de pré-processamento e coeficientes do modelo
- NOVO: salva scaler e modelo com joblib + PDF resumo (1 página)

Uso:
    python gerar_evidencias_heart.py --csv heart.csv --target HeartDisease --outdir evidencias
"""
import argparse
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    precision_recall_fscore_support,
    accuracy_score,
    recall_score
)
import joblib
import matplotlib

# PDF resumo
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def make_pdf_resumo(pdf_path, header, metrics_dict, cm_png, roc_png):
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    w, h = A4
    y = h - 2*cm

    # Título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, y, "Resumo de Evidências – Risco Cardíaco (Logistic Regression)")
    y -= 0.8*cm
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, header)
    y -= 0.8*cm

    # Métricas
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Métricas (teste):")
    y -= 0.6*cm
    c.setFont("Helvetica", 11)
    for k in ["accuracy","precision","recall","f1","auc"]:
        if k in metrics_dict:
            c.drawString(2*cm, y, f"- {k.upper()}: {metrics_dict[k]:.4f}")
            y -= 0.5*cm

    # Imagens (lado a lado se couber, senão empilhadas)
    img_w = (w - 4*cm) / 2 - 0.5*cm
    img_h = 6*cm

    # Ajuste vertical
    if y < 12*cm:
        c.showPage()
        y = h - 2*cm

    if Path(cm_png).exists():
        c.drawImage(str(cm_png), 2*cm, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
    if Path(roc_png).exists():
        c.drawImage(str(roc_png), 2*cm + img_w + 1*cm, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')

    c.showPage()
    c.save()

def main():
    parser = argparse.ArgumentParser(description="Gerar evidências do modelo cardíaco (Logistic Regression).")
    parser.add_argument("--csv", type=str, default="heart.csv", help="Caminho para o dataset (CSV).")
    parser.add_argument("--target", type=str, default="HeartDisease", help="Nome da coluna alvo.")
    parser.add_argument("--outdir", type=str, default="evidencias", help="Diretório de saída.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path.resolve()}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    # -------------------- Carregamento --------------------
    df = pd.read_csv(csv_path)
    if args.target not in df.columns:
        raise ValueError(f"Coluna alvo '{args.target}' não encontrada nas colunas: {list(df.columns)}")

    # -------------------- Checagem de nulos --------------------
    nulls = df.isnull().sum().sort_values(ascending=False)

    # -------------------- Balanceamento --------------------
    class_balance = df[args.target].value_counts(dropna=False).to_frame("count")
    class_balance["pct"] = (class_balance["count"] / class_balance["count"].sum()).round(4)

    # -------------------- One-Hot --------------------
    df_encoded = pd.get_dummies(df, drop_first=True)
    # Define X e y, garantindo que y seja 0/1 e X não contenha a coluna target
    y = df[args.target].astype(int)
    X = df_encoded.drop(columns=[c for c in df_encoded.columns if c == args.target or c.startswith(args.target+"_") or c.startswith(args.target)])

    # -------------------- Split 70/30 com stratify --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # -------------------- Escalonamento --------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------- Treinamento --------------------
    model = LogisticRegression(solver="liblinear")
    model.fit(X_train_scaled, y_train)

    # -------------------- Predição e Métricas --------------------
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    report_txt = classification_report(y_test, y_pred, digits=4)

    # -------------------- Evidências Visuais --------------------
    matplotlib.use("Agg")  # garante render sem display
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão - Logistic Regression")
    plt.xlabel("Predito"); plt.ylabel("Verdadeiro")
    cm_path = outdir / f"matriz_confusao_{ts}.png"
    plt.tight_layout(); plt.savefig(cm_path, dpi=140); plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("Curva ROC - Logistic Regression")
    roc_path = outdir / f"roc_curve_{ts}.png"
    plt.tight_layout(); plt.savefig(roc_path, dpi=140); plt.close()

    # -------------------- Coeficientes --------------------
    coef_series = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
    coef_csv = outdir / f"coeficientes_{ts}.csv"
    coef_series.to_csv(coef_csv, header=["coeficiente"])

    # -------------------- Persistência do Modelo/Scaler --------------------
    model_path = outdir / f"modelo_logreg_{ts}.joblib"
    scaler_path = outdir / f"scaler_standard_{ts}.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # salvar nomes das features (importante para alinhamento futuro)
    features_json = outdir / f"features_{ts}.json"
    with open(features_json, "w", encoding="utf-8") as f:
        json.dump({"features": list(X.columns)}, f, indent=2, ensure_ascii=False)

    # -------------------- Logs Texto --------------------
    preprocess_log = outdir / f"evidencias_preprocess_{ts}.txt"
    with open(preprocess_log, "w", encoding="utf-8") as f:
        f.write("=== Checagem de nulos ===\n")
        f.write(nulls.to_string()); f.write("\n\n")
        f.write("=== Balanceamento da classe alvo ===\n")
        f.write(class_balance.to_string()); f.write("\n\n")
        f.write(f"Shape original: {df.shape}\n")
        f.write(f"Shape após One-Hot: {df_encoded.shape}\n")
        f.write(f"Treino: {X_train.shape}, Teste: {X_test.shape}\n")

    report_path = outdir / f"classification_report_{ts}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    # -------------------- Saída JSON consolidada --------------------
    resultados = {
        "timestamp": ts,
        "csv": str(csv_path),
        "target": args.target,
        "amostras_treino": int(X_train.shape[0]),
        "amostras_teste": int(X_test.shape[0]),
        "features": list(X.columns),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc)
        },
        "artifacts": {
            "matriz_confusao_png": str(cm_path),
            "roc_curve_png": str(roc_path),
            "classification_report_txt": str(report_path),
            "preprocess_log_txt": str(preprocess_log),
            "coeficientes_csv": str(coef_csv),
            "model_joblib": str(model_path),
            "scaler_joblib": str(scaler_path),
            "features_json": str(features_json),
        }
    }
    json_path = outdir / f"evidencias_treinamento_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # -------------------- PDF Resumo --------------------
    header = f"CSV: {csv_path.name} | Target: {args.target} | Train/Test: 70/30 | Data: {ts}"
    pdf_path = outdir / f"resumo_evidencias_{ts}.pdf"
    make_pdf_resumo(pdf_path, header, resultados["metrics"], cm_path, roc_path)

    print("\n✅ Evidências geradas com sucesso!")
    print(f"- JSON métricas: {json_path}")
    print(f"- Relatório:     {report_path}")
    print(f"- Matriz conf.:  {cm_path}")
    print(f"- ROC curve:     {roc_path}")
    print(f"- Coeficientes:  {coef_csv}")
    print(f"- Modelo:        {model_path}")
    print(f"- Scaler:        {scaler_path}")
    print(f"- PDF Resumo:    {pdf_path}")
    print(f"- Pré-processo:  {preprocess_log}")
    print("\nDica: inclua esses arquivos no relatório e nos slides.")

if __name__ == "__main__":
    main()
