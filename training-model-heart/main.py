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
import logging
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

from ag_algoritmo import AlgoritmoGenetico, executar_ag
from ag_experimentos import executar_multiplos_experimentos, gerar_relatorio_comparativo, definir_experimentos_padrao
from ag_comparacao import treinar_modelo_original, comparar_modelos, imprimir_comparacao
from ag_logging import configurar_logger

def make_pdf_resumo(pdf_path, header, metrics_dict, cm_png, roc_png, experimentos_executados=None):
    """
    #AG Gera PDF resumo com métricas e gráficos.
    Se experimentos_executados for fornecido, inclui gráficos de todos os experimentos.
    """
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

    # #AG Se há múltiplos experimentos, inclui todos os gráficos
    if experimentos_executados and len(experimentos_executados) > 1:
        # Nova página para gráficos dos experimentos
        c.showPage()
        y = h - 2*cm
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, y, "Gráficos dos Experimentos (Modelos Otimizados pelo AG)")
        y -= 1*cm
        
        # Dimensões dos gráficos (lado a lado: CM e ROC)
        img_w = (w - 4*cm) / 2 - 0.5*cm
        img_h = 5.5*cm
        
        for i, exp in enumerate(experimentos_executados, 1):
            # Nova página para cada experimento (melhor organização)
            c.showPage()
            y = h - 2*cm
            
            # Título do experimento
            c.setFont("Helvetica-Bold", 12)
            nome_exp = exp.nome_experimento[:60]  # Trunca se muito longo
            c.drawString(2*cm, y, f"Experimento {i}: {nome_exp}")
            y -= 0.8*cm
            
            # Métricas do modelo otimizado
            if exp.comparacao and 'modelo_otimizado' in exp.comparacao:
                c.setFont("Helvetica", 10)
                metrics = exp.comparacao['modelo_otimizado']
                c.drawString(2*cm, y, f"Accuracy: {metrics.get('accuracy', 0):.4f} | "
                                     f"Recall: {metrics.get('recall', 0):.4f} | "
                                     f"F1: {metrics.get('f1', 0):.4f} | "
                                     f"AUC: {metrics.get('auc', 0):.4f}")
                y -= 0.8*cm
            
            # Posições dos gráficos (lado a lado)
            y_img = y - img_h
            
            # Matriz de Confusão (esquerda)
            if exp.cm_png_path and Path(exp.cm_png_path).exists():
                c.drawImage(str(exp.cm_png_path), 2*cm, y_img, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
                c.setFont("Helvetica", 9)
                c.drawString(2*cm, y_img - 0.4*cm, "Matriz de Confusão")
            
            # Curva ROC (direita)
            if exp.roc_png_path and Path(exp.roc_png_path).exists():
                c.drawImage(str(exp.roc_png_path), 2*cm + img_w + 1*cm, y_img, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
                c.setFont("Helvetica", 9)
                c.drawString(2*cm + img_w + 1*cm, y_img - 0.4*cm, "Curva ROC")
        
        c.showPage()
    else:
        # Modo original: apenas um par de gráficos (lado a lado)
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
    parser = argparse.ArgumentParser(description="Gerar evidências do modelo cardíaco (Logistic Regression) com AG.")
    parser.add_argument("--csv", type=str, default="heart.csv", help="Caminho para o dataset (CSV).")
    parser.add_argument("--target", type=str, default="HeartDisease", help="Nome da coluna alvo.")
    parser.add_argument("--outdir", type=str, default="evidencias", help="Diretório de saída.")
    parser.add_argument("--modo", type=str, default="experimentos", 
                       choices=["simples", "experimentos", "comparar"],
                       help="Modo de execução: 'simples' (um AG), 'experimentos' (múltiplos experimentos), 'comparar' (um AG + comparação).")
    parser.add_argument("--num_experimentos", type=int, default=3,
                       help="Número de experimentos a executar (apenas no modo 'experimentos').")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path.resolve()}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # #AG Configura logging estruturado para todo o pipeline
    logger_principal = configurar_logger(
        nome_logger='AG.Main',
        diretorio_logs=str(outdir / 'logs'),
        nivel=logging.INFO,
        salvar_arquivo=True,
        formato_detalhado=True
    )
    logger_principal.info(f"#AG Iniciando pipeline de treinamento - Modo: {args.modo}")
    logger_principal.info(f"#AG Dataset: {args.csv}, Target: {args.target}")

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
    
    # #AG Garante que y_train e y_test são arrays numpy (não Series do pandas)
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    # -------------------- Algoritmo Genético e Comparação --------------------
    # #AG Um Algoritmo Genético foi empregado para otimização dos hiperparâmetros do modelo de Regressão Logística.
    # #AG Cada indivíduo da população representa um conjunto de hiperparâmetros (C, penalty, class_weight, max_iter).
    # #AG A função fitness foi definida como combinação ponderada de accuracy (20%), recall (30%), F1-score (25%) e AUC (25%),
    # #AG obtida por validação cruzada estratificada de 5 folds. Após a convergência do algoritmo genético,
    # #AG o melhor conjunto de hiperparâmetros foi utilizado para o treinamento final do modelo.
    
    melhor_individuo = None
    model = None
    y_pred = None
    y_prob = None
    experimentos_executados = None  # #AG Inicializa para uso no PDF
    
    if args.modo == "experimentos":
        # #AG Modo: Executa múltiplos experimentos com diferentes configurações do AG
        print("\n#AG Modo: MÚLTIPLOS EXPERIMENTOS")
        print("#AG " + "="*80)
        
        # Define experimentos (pelo menos 3, conforme solicitado)
        experimentos_config = definir_experimentos_padrao()
        if args.num_experimentos < len(experimentos_config):
            experimentos_config = experimentos_config[:args.num_experimentos]
        
        # Executa múltiplos experimentos
        experimentos_executados = executar_multiplos_experimentos(
            X_train_scaled, y_train, X_test_scaled, y_test,
            experimentos_config=experimentos_config,
            verbose=True,
            outdir=outdir,
            timestamp=ts
        )
        
        # Gera relatório comparativo
        relatorio_path = gerar_relatorio_comparativo(experimentos_executados, outdir, ts)
        
        # Usa o melhor modelo do melhor experimento (por AUC) para continuar o pipeline
        melhor_experimento = max(experimentos_executados, 
                                 key=lambda e: e.comparacao['modelo_otimizado']['auc'])
        model = melhor_experimento.modelo_otimizado
        melhor_individuo = melhor_experimento.melhor_individuo
        
        print(f"\n#AG Usando melhor modelo do experimento: {melhor_experimento.nome_experimento}")
        
    elif args.modo == "comparar":
        # #AG Modo: Executa um AG e compara com modelo original
        print("\n#AG Modo: COMPARAÇÃO (AG vs Original)")
        print("#AG " + "="*80)
        
        # Executa o algoritmo genético
        melhor_individuo = executar_ag(
            X_train_scaled, y_train,
            tamanho_populacao=20,
            n_geracoes=15,
            taxa_cruzamento=0.7,
            taxa_mutacao=0.1,
            n_elites=2,
            metodo_selecao='torneio',
            metodo_cruzamento='uniforme',
            metodo_mutacao='uniforme',
            metric='composite',
            cv_folds=5,
            verbose=True
        )
        
        # Treina modelo original (padrão)
        modelo_original = treinar_modelo_original(X_train_scaled, y_train)
        
        # Treina modelo otimizado
        model = LogisticRegression(
            solver="liblinear",
            C=melhor_individuo["C"],
            penalty=melhor_individuo["penalty"],
            class_weight=melhor_individuo["class_weight"],
            max_iter=melhor_individuo["max_iter"],
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Compara modelos
        comparacao = comparar_modelos(modelo_original, model, X_test_scaled, y_test)
        imprimir_comparacao(comparacao, melhor_individuo)
        
    else:  # modo "simples"
        # #AG Modo: Executa um único AG (comportamento padrão original)
        print("\n#AG Modo: SIMPLES (Um único AG)")
        print("#AG " + "="*80)
        
        melhor_individuo = executar_ag(
            X_train_scaled, y_train,
            tamanho_populacao=20,
            n_geracoes=15,
            taxa_cruzamento=0.7,
            taxa_mutacao=0.1,
            n_elites=2,
            metodo_selecao='torneio',
            metodo_cruzamento='uniforme',
            metodo_mutacao='uniforme',
            metric='composite',
            cv_folds=5,
            verbose=True
        )
        
        # Treina modelo otimizado
        model = LogisticRegression(
            solver="liblinear",
            C=melhor_individuo["C"],
            penalty=melhor_individuo["penalty"],
            class_weight=melhor_individuo["class_weight"],
            max_iter=melhor_individuo["max_iter"],
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

    # -------------------- Predição e Métricas --------------------
    if y_pred is None:  # Se ainda não foi calculado (modos simples/comparar)
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
        "modo": args.modo,
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
    
    # #AG Adiciona informações do algoritmo genético se aplicável
    if melhor_individuo:
        resultados["algoritmo_genetico"] = {
            "hiperparametros_otimizados": {
                "C": float(melhor_individuo["C"]),
                "penalty": melhor_individuo["penalty"],
                "class_weight": str(melhor_individuo["class_weight"]),
                "max_iter": int(melhor_individuo["max_iter"])
            }
        }
    json_path = outdir / f"evidencias_treinamento_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # -------------------- PDF Resumo --------------------
    header = f"CSV: {csv_path.name} | Target: {args.target} | Train/Test: 70/30 | Data: {ts}"
    pdf_path = outdir / f"resumo_evidencias_{ts}.pdf"
    
    # #AG Se está no modo experimentos, passa a lista de experimentos para incluir todos os gráficos
    make_pdf_resumo(pdf_path, header, resultados["metrics"], cm_path, roc_path, experimentos_executados)

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
