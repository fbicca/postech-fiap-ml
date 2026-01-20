#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#AG Módulo para gerenciar múltiplos experimentos com diferentes configurações do algoritmo genético.
"""

import json
import datetime as dt
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Para evitar problemas de display
import matplotlib.pyplot as plt
import seaborn as sns
from ag_algoritmo import AlgoritmoGenetico
from ag_comparacao import treinar_modelo_original, comparar_modelos, avaliar_modelo
from ag_logging import criar_logger_por_modulo
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, RocCurveDisplay

# #AG Logger estruturado para experimentos
logger_experimentos = criar_logger_por_modulo('ag_experimentos')


class ExperimentoAG:
    """
    #AG Classe para executar um único experimento com configuração específica do AG.
    """
    
    def __init__(
        self,
        nome_experimento,
        X_train, y_train, X_test, y_test,
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
        outdir=None,
        timestamp=None
    ):
        """
        #AG Inicializa um experimento com configurações específicas.
        
        Args:
            nome_experimento (str): Nome identificador do experimento
            X_train, y_train, X_test, y_test: Dados de treino e teste
            ... (outros parâmetros do AG): Configurações do algoritmo genético
        """
        self.nome_experimento = nome_experimento
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Configurações do AG
        self.tamanho_populacao = tamanho_populacao
        self.n_geracoes = n_geracoes
        self.taxa_cruzamento = taxa_cruzamento
        self.taxa_mutacao = taxa_mutacao
        self.n_elites = n_elites
        self.metodo_selecao = metodo_selecao
        self.metodo_cruzamento = metodo_cruzamento
        self.metodo_mutacao = metodo_mutacao
        self.metric = metric
        self.cv_folds = cv_folds
        
        # Resultados (preenchidos após execução)
        self.melhor_individuo = None
        self.modelo_otimizado = None
        self.modelo_original = None
        self.comparacao = None
        self.historico_ag = None
        
        # #AG Caminhos dos gráficos (preenchidos após gerar gráficos)
        self.outdir = outdir
        self.timestamp = timestamp
        self.cm_png_path = None
        self.roc_png_path = None
    
    def executar(self, verbose=True):
        """
        #AG Executa o experimento completo: AG + treinamento + comparação.
        
        Args:
            verbose (bool): Se True, imprime progresso
        """
        if verbose:
            print(f"\n#AG {'='*80}")
            print(f"#AG EXPERIMENTO: {self.nome_experimento}")
            print(f"#AG {'='*80}")
            print(f"#AG Configuração do AG:")
            print(f"#AG   População: {self.tamanho_populacao}")
            print(f"#AG   Gerações: {self.n_geracoes}")
            print(f"#AG   Taxa Cruzamento: {self.taxa_cruzamento}")
            print(f"#AG   Taxa Mutação: {self.taxa_mutacao}")
            print(f"#AG   Elites: {self.n_elites}")
            print(f"#AG   Seleção: {self.metodo_selecao}")
            print(f"#AG   Cruzamento: {self.metodo_cruzamento}")
            print(f"#AG   Mutação: {self.metodo_mutacao}")
            print(f"#AG   Métrica: {self.metric}")
        
        # #AG Executa o algoritmo genético com logging e salvamento de histórico
        ag = AlgoritmoGenetico(
            tamanho_populacao=self.tamanho_populacao,
            n_geracoes=self.n_geracoes,
            taxa_cruzamento=self.taxa_cruzamento,
            taxa_mutacao=self.taxa_mutacao,
            n_elites=self.n_elites,
            metodo_selecao=self.metodo_selecao,
            metodo_cruzamento=self.metodo_cruzamento,
            metodo_mutacao=self.metodo_mutacao,
            metric=self.metric,
            cv_folds=self.cv_folds,
            verbose=verbose,
            salvar_historico=True,
            diretorio_logs='logs',
            gerar_graficos=True
        )
        
        self.melhor_individuo = ag.evoluir(self.X_train, self.y_train, nome_experimento=self.nome_experimento)
        self.historico_ag = ag.get_historico()
        
        # #AG Log de tempo de execução
        tempo_exec = ag.get_tempo_execucao()
        if verbose:
            logger_experimentos.info(f"#AG Tempo de execução do AG: {tempo_exec['duracao_total']:.2f}s")
        
        # Treina modelos
        if verbose:
            print("\n#AG Treinando modelos...")
        
        # Modelo original (padrão)
        self.modelo_original = treinar_modelo_original(self.X_train, self.y_train)
        
        # Modelo otimizado (com hiperparâmetros do AG)
        self.modelo_otimizado = LogisticRegression(
            solver='liblinear',
            C=self.melhor_individuo['C'],
            penalty=self.melhor_individuo['penalty'],
            class_weight=self.melhor_individuo['class_weight'],
            max_iter=self.melhor_individuo['max_iter'],
            random_state=42
        )
        self.modelo_otimizado.fit(self.X_train, self.y_train)
        
        # Compara modelos
        self.comparacao = comparar_modelos(
            self.modelo_original,
            self.modelo_otimizado,
            self.X_test,
            self.y_test
        )
        
        # #AG Gera gráficos de matriz de confusão e ROC para este experimento
        if self.outdir and self.timestamp:
            self._gerar_graficos_experimento()
        
        if verbose:
            from ag_comparacao import imprimir_comparacao
            imprimir_comparacao(self.comparacao, self.melhor_individuo)
    
    def _gerar_graficos_experimento(self):
        """
        #AG Gera e salva gráficos de matriz de confusão e ROC para o modelo otimizado deste experimento.
        """
        if self.modelo_otimizado is None:
            return
        
        # Cria diretório se não existir
        outdir_path = Path(self.outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        
        # Nome do experimento limpo para usar no nome do arquivo
        nome_limpo = self.nome_experimento.replace(' ', '_').replace(':', '').replace('/', '_').replace('(', '').replace(')', '').lower()
        
        # Predições
        y_pred = self.modelo_otimizado.predict(self.X_test)
        y_prob = self.modelo_otimizado.predict_proba(self.X_test)[:, 1]
        
        # Matriz de Confusão
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusão - {self.nome_experimento}")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        self.cm_png_path = outdir_path / f"matriz_confusao_{nome_limpo}_{self.timestamp}.png"
        plt.tight_layout()
        plt.savefig(self.cm_png_path, dpi=140)
        plt.close()
        
        # Curva ROC
        RocCurveDisplay.from_predictions(self.y_test, y_prob)
        plt.title(f"Curva ROC - {self.nome_experimento}")
        self.roc_png_path = outdir_path / f"roc_curve_{nome_limpo}_{self.timestamp}.png"
        plt.tight_layout()
        plt.savefig(self.roc_png_path, dpi=140)
        plt.close()
        
        logger_experimentos.info(f"#AG Gráficos do experimento '{self.nome_experimento}' salvos:")
        logger_experimentos.info(f"#AG   - Matriz de Confusão: {self.cm_png_path}")
        logger_experimentos.info(f"#AG   - ROC: {self.roc_png_path}")
    
    def get_resultado_dict(self):
        """
        #AG Retorna os resultados do experimento em formato de dicionário.
        
        Returns:
            dict: Dicionário com todos os resultados
        """
        return {
            'nome_experimento': self.nome_experimento,
            'configuracao_ag': {
                'tamanho_populacao': self.tamanho_populacao,
                'n_geracoes': self.n_geracoes,
                'taxa_cruzamento': self.taxa_cruzamento,
                'taxa_mutacao': self.taxa_mutacao,
                'n_elites': self.n_elites,
                'metodo_selecao': self.metodo_selecao,
                'metodo_cruzamento': self.metodo_cruzamento,
                'metodo_mutacao': self.metodo_mutacao,
                'metric': self.metric,
                'cv_folds': self.cv_folds
            },
            'hiperparametros_otimizados': {
                'C': float(self.melhor_individuo['C']),
                'penalty': self.melhor_individuo['penalty'],
                'class_weight': str(self.melhor_individuo['class_weight']),
                'max_iter': int(self.melhor_individuo['max_iter'])
            },
            'comparacao': self.comparacao,
            'historico_ag': self.historico_ag
        }


def definir_experimentos_padrao():
    """
    #AG Define um conjunto padrão de 3+ experimentos com diferentes configurações do AG.
    
    Returns:
        list: Lista de dicionários com configurações de experimentos
    """
    experimentos = [
        {
            'nome': 'Experimento 1: Configuração Conservadora',
            'tamanho_populacao': 20,
            'n_geracoes': 15,
            'taxa_cruzamento': 0.7,
            'taxa_mutacao': 0.1,
            'n_elites': 2,
            'metodo_selecao': 'torneio',
            'metodo_cruzamento': 'uniforme',
            'metodo_mutacao': 'uniforme',
            'metric': 'composite'
        },
        {
            'nome': 'Experimento 2: População Maior + Mais Gerações',
            'tamanho_populacao': 40,
            'n_geracoes': 25,
            'taxa_cruzamento': 0.7,
            'taxa_mutacao': 0.1,
            'n_elites': 4,
            'metodo_selecao': 'torneio',
            'metodo_cruzamento': 'uniforme',
            'metodo_mutacao': 'uniforme',
            'metric': 'composite'
        },
        {
            'nome': 'Experimento 3: Alta Mutação + Cruzamento Aritmético',
            'tamanho_populacao': 30,
            'n_geracoes': 20,
            'taxa_cruzamento': 0.8,
            'taxa_mutacao': 0.2,
            'n_elites': 3,
            'metodo_selecao': 'roleta',
            'metodo_cruzamento': 'aritmetico',
            'metodo_mutacao': 'gaussiana',
            'metric': 'composite'
        },
        {
            'nome': 'Experimento 4: Foco em RECALL (Sensibilidade)',
            #AG Configuração desenhada para priorizar Recall com boa exploração,
            #AG mas sem população tão grande (compromisso entre custo e desempenho).
            'tamanho_populacao': 30,
            'n_geracoes': 25,
            'taxa_cruzamento': 0.8,
            'taxa_mutacao': 0.15,
            'n_elites': 3,
            'metodo_selecao': 'torneio',
            'metodo_cruzamento': 'aritmetico',   # mistura suave de C e max_iter
            'metodo_mutacao': 'nao_uniforme',    # mutação forte no início, suave no fim
            'metric': 'recall'                   # prioriza sensibilidade (poucos falsos negativos)
        }
    ]
    
    return experimentos


def executar_multiplos_experimentos(X_train, y_train, X_test, y_test, experimentos_config=None, verbose=True, outdir=None, timestamp=None):
    """
    #AG Executa múltiplos experimentos com diferentes configurações do AG.
    
    Args:
        X_train, y_train, X_test, y_test: Dados de treino e teste
        experimentos_config (list): Lista de configurações de experimentos. Se None, usa padrão.
        verbose (bool): Se True, imprime progresso
        outdir (str ou Path): Diretório para salvar gráficos e logs
        timestamp (str): Timestamp para usar nos nomes dos arquivos
        
    Returns:
        list: Lista de objetos ExperimentoAG executados
    """
    if experimentos_config is None:
        experimentos_config = definir_experimentos_padrao()
    
    experimentos_executados = []
    
    if verbose:
        print("\n#AG " + "="*80)
        print(f"#AG EXECUTANDO {len(experimentos_config)} EXPERIMENTOS")
        print("#AG " + "="*80)
    
    for i, config in enumerate(experimentos_config, 1):
        if verbose:
            print(f"\n#AG [{i}/{len(experimentos_config)}] Preparando experimento...")
        
        experimento = ExperimentoAG(
            nome_experimento=config['nome'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tamanho_populacao=config['tamanho_populacao'],
            n_geracoes=config['n_geracoes'],
            taxa_cruzamento=config['taxa_cruzamento'],
            taxa_mutacao=config['taxa_mutacao'],
            n_elites=config['n_elites'],
            metodo_selecao=config['metodo_selecao'],
            metodo_cruzamento=config['metodo_cruzamento'],
            metodo_mutacao=config['metodo_mutacao'],
            metric=config.get('metric', 'composite'),
            cv_folds=5,
            outdir=outdir,
            timestamp=timestamp
        )
        
        experimento.executar(verbose=verbose)
        experimentos_executados.append(experimento)
    
    return experimentos_executados


def gerar_relatorio_comparativo(experimentos_executados, outdir, timestamp=None):
    """
    #AG Gera relatório comparativo de todos os experimentos.
    
    Args:
        experimentos_executados (list): Lista de objetos ExperimentoAG
        outdir (Path): Diretório de saída
        timestamp (str): Timestamp para nomeação dos arquivos
        
    Returns:
        Path: Caminho do arquivo JSON gerado
    """
    if timestamp is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Coleta resultados de todos os experimentos
    resultados_gerais = {
        'timestamp': timestamp,
        'numero_experimentos': len(experimentos_executados),
        'experimentos': [exp.get_resultado_dict() for exp in experimentos_executados]
    }
    
    # Análise comparativa entre experimentos
    analise = {
        'melhor_accuracy': None,
        'melhor_recall': None,
        'melhor_f1': None,
        'melhor_auc': None,
        'experimentos_ordenados_por_metricas': {}
    }
    
    if experimentos_executados:
        # Encontra melhor experimento para cada métrica
        best_acc_exp = max(experimentos_executados, 
                          key=lambda e: e.comparacao['modelo_otimizado']['accuracy'])
        best_rec_exp = max(experimentos_executados, 
                          key=lambda e: e.comparacao['modelo_otimizado']['recall'])
        best_f1_exp = max(experimentos_executados, 
                         key=lambda e: e.comparacao['modelo_otimizado']['f1'])
        best_auc_exp = max(experimentos_executados, 
                          key=lambda e: e.comparacao['modelo_otimizado']['auc'])
        
        analise['melhor_accuracy'] = {
            'experimento': best_acc_exp.nome_experimento,
            'valor': float(best_acc_exp.comparacao['modelo_otimizado']['accuracy'])
        }
        analise['melhor_recall'] = {
            'experimento': best_rec_exp.nome_experimento,
            'valor': float(best_rec_exp.comparacao['modelo_otimizado']['recall'])
        }
        analise['melhor_f1'] = {
            'experimento': best_f1_exp.nome_experimento,
            'valor': float(best_f1_exp.comparacao['modelo_otimizado']['f1'])
        }
        analise['melhor_auc'] = {
            'experimento': best_auc_exp.nome_experimento,
            'valor': float(best_auc_exp.comparacao['modelo_otimizado']['auc'])
        }
        
        # Ordena experimentos por cada métrica
        analise['experimentos_ordenados_por_metricas'] = {
            'accuracy': sorted([(e.nome_experimento, e.comparacao['modelo_otimizado']['accuracy']) 
                               for e in experimentos_executados], 
                              key=lambda x: x[1], reverse=True),
            'recall': sorted([(e.nome_experimento, e.comparacao['modelo_otimizado']['recall']) 
                             for e in experimentos_executados], 
                            key=lambda x: x[1], reverse=True),
            'f1': sorted([(e.nome_experimento, e.comparacao['modelo_otimizado']['f1']) 
                         for e in experimentos_executados], 
                        key=lambda x: x[1], reverse=True),
            'auc': sorted([(e.nome_experimento, e.comparacao['modelo_otimizado']['auc']) 
                          for e in experimentos_executados], 
                         key=lambda x: x[1], reverse=True)
        }
    
    resultados_gerais['analise_comparativa'] = analise
    
    # Salva JSON
    json_path = outdir / f"comparacao_experimentos_ag_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(resultados_gerais, f, indent=2, ensure_ascii=False)
    
    # Imprime resumo
    print("\n#AG " + "="*80)
    print("#AG RELATÓRIO COMPARATIVO DE EXPERIMENTOS")
    print("#AG " + "="*80)
    print(f"\n#AG Melhores resultados por métrica:")
    print(f"#AG   Accuracy: {analise['melhor_accuracy']['experimento']} ({analise['melhor_accuracy']['valor']:.4f})")
    print(f"#AG   Recall: {analise['melhor_recall']['experimento']} ({analise['melhor_recall']['valor']:.4f})")
    print(f"#AG   F1-Score: {analise['melhor_f1']['experimento']} ({analise['melhor_f1']['valor']:.4f})")
    print(f"#AG   AUC: {analise['melhor_auc']['experimento']} ({analise['melhor_auc']['valor']:.4f})")
    print(f"\n#AG Relatório completo salvo em: {json_path}")
    print("#AG " + "="*80 + "\n")
    
    return json_path

