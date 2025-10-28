# Atividade Prática: Classificação Supervisionada
# Algoritmos: Árvore de Decisão e KNN
# Autor: Larissa Campos Cardoso
# Data: 28/10/2025

# Imports

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score
)
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("ATIVIDADE PRÁTICA: CLASSIFICAÇÃO SUPERVISIONADA")
print("Algoritmos: Árvore de Decisão e KNN")
print("=" * 80)

# 2. CARREGAMENTO DADOS

print("\n" + "=" * 80)
print("2. CARREGAMENTO E EXPLORAÇÃO DOS DADOS")
print("=" * 80)

# Carregar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

print("\nDataset: IRIS")
print(f"   Dimensoes: {X.shape}")
print(f"   Numero de amostras: {X.shape[0]}")
print(f"   Numero de features: {X.shape[1]}")
print(f"   Classes: {iris.target_names.tolist()}")
print(f"   Features: {iris.feature_names}")

# Estatísticas básicas
print("\nDistribuicao das classes:")
unique, counts = np.unique(y, return_counts=True)
for i, (cls, count) in enumerate(zip(unique, counts)):
    print(f"   {iris.target_names[cls]}: {count} amostras")

print("\nEstatisticas das features:")
for i, feature in enumerate(iris.feature_names):
    print(f"   {feature}:")
    print(f"      Média: {np.mean(X[:, i]):.2f}")
    print(f"      Desvio padrão: {np.std(X[:, i]):.2f}")
    print(f"      Min: {np.min(X[:, i]):.2f}, Max: {np.max(X[:, i]):.2f}")

# 3. DIVISÃO DOS DADOS

print("\n" + "=" * 80)
print("3. DIVISÃO DOS DADOS")
print("=" * 80)

# treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

print(f"\n[OK] Conjunto de treino: {X_train.shape[0]} amostras")
print(f"[OK] Conjunto de teste: {X_test.shape[0]} amostras")
print(f"[OK] Proporcao: 70% treino / 30% teste")

# Normalização (importante para KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[OK] Dados normalizados (StandardScaler)")

# 4. MODELO 1: ÁRVORE DE DECISÃO

print("\n" + "=" * 80)
print("4. MODELO 1: ÁRVORE DE DECISÃO")
print("=" * 80)

# Treinamento
dt_model = DecisionTreeClassifier(
    max_depth=4, 
    random_state=42,
    criterion='gini'
)
dt_model.fit(X_train, y_train)

print("\n[OK] Modelo treinado com sucesso!")
print(f"  Profundidade da arvore: {dt_model.get_depth()}")
print(f"  Numero de folhas: {dt_model.get_n_leaves()}")

# Predições
y_pred_dt = dt_model.predict(X_test)

# Métricas
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt, average='weighted')
dt_recall = recall_score(y_test, y_pred_dt, average='weighted')
dt_f1 = f1_score(y_test, y_pred_dt, average='weighted')

print("\nMETRIcAS - ARVORE DE DECISAO:")
print(f"   Acuracia:  {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")
print(f"   Precisao:  {dt_precision:.4f}")
print(f"   Recall:    {dt_recall:.4f}")
print(f"   F1-Score:  {dt_f1:.4f}")

# Matriz de Confusão
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("\nMatriz de Confusao:")
print("    Predito →")
print(f"Real ↓  {iris.target_names[0]:12} {iris.target_names[1]:12} {iris.target_names[2]:12}")
for i, row in enumerate(cm_dt):
    print(f"{iris.target_names[i]:8} {row[0]:12} {row[1]:12} {row[2]:12}")

# Relatório de classificação
print("\nRelatorio de Classificacao:")
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))

# 5. MODELO 2: KNN 

print("\n" + "=" * 80)
print("5. MODELO 2: KNN (K-NEAREST NEIGHBORS)")
print("=" * 80)

# Treinamento com k=5
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train_scaled, y_train)

print("\n[OK] Modelo KNN treinado com sucesso!")
print(f"  Numero de vizinhos (k): {knn_model.n_neighbors}")
print(f"  Metrica de distancia: {knn_model.metric}")

# Predições
y_pred_knn = knn_model.predict(X_test_scaled)

# Métricas
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn, average='weighted')
knn_recall = recall_score(y_test, y_pred_knn, average='weighted')
knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')

print("\nMETRICAS - KNN:")
print(f"   Acuracia:  {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
print(f"   Precisao:  {knn_precision:.4f}")
print(f"   Recall:    {knn_recall:.4f}")
print(f"   F1-Score:  {knn_f1:.4f}")

# Matriz de Confusão
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("\nMatriz de Confusao:")
print("    Predito →")
print(f"Real ↓  {iris.target_names[0]:12} {iris.target_names[1]:12} {iris.target_names[2]:12}")
for i, row in enumerate(cm_knn):
    print(f"{iris.target_names[i]:8} {row[0]:12} {row[1]:12} {row[2]:12}")

# Relatório de classificação
print("\nRelatorio de Classificacao:")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

# 6. ANÁLISE KNN

print("\n" + "=" * 80)
print("6. ANALISE DE DIFERENTES VALORES DE K")
print("=" * 80)

k_values = range(1, 21)
k_scores = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    score = knn_temp.score(X_test_scaled, y_test)
    k_scores.append(score)

print("\nAcuracia por valor de K:")
print("K  | Acuracia")
print("---|----------")
for k, score in zip(k_values, k_scores):
    marker = " <- MELHOR" if score == max(k_scores) else ""
    print(f"{k:2d} | {score:.4f}{marker}")

best_k = k_values[np.argmax(k_scores)]
best_score = max(k_scores)
print(f"\n[OK] Melhor valor de k: {best_k} (Acuracia: {best_score:.4f})")

# 7. ANÁLISE FINAL

print("\n" + "=" * 80)
print("7. ANALISE COMPARATIVA FINAL")
print("=" * 80)

print("\n" + "="*80)
print("RESUMO COMPARATIVO DOS MODELOS")
print("="*80)

print("\nMétrica          | Árvore de Decisão | KNN (k=5)")
print("-----------------|-------------------|----------")
print(f"Acurácia         | {dt_accuracy:17.4f} | {knn_accuracy:.4f}")
print(f"Precisão         | {dt_precision:17.4f} | {knn_precision:.4f}")
print(f"Recall           | {dt_recall:17.4f} | {knn_recall:.4f}")
print(f"F1-Score         | {dt_f1:17.4f} | {knn_f1:.4f}")

# Determinar melhor modelo
if dt_f1 > knn_f1:
    melhor = "Árvore de Decisão"
    diferenca = dt_f1 - knn_f1
else:
    melhor = "KNN"
    diferenca = knn_f1 - dt_f1

print(f"\n*** MELHOR MODELO: {melhor}")
print(f"   Diferenca de F1-Score: {diferenca:.4f}")

# 8. CONCLUSÕES 

print("\n" + "=" * 80)
print("8. CONCLUSOES E INTERPRETACOES")
print("=" * 80)

print("""
ANALISE FINAL:

1. DESEMPENHO GERAL:
   Ambos os modelos apresentaram excelente desempenho no dataset Iris,
   com acuracias superiores a 95%. Isso se deve a natureza bem separavel
   das classes neste dataset classico.

2. ARVORE DE DECISAO:
   [+] Vantagens:
     - Interpretavel e visual (podemos ver a logica de decisao)
     - Nao requer normalizacao dos dados
     - Rapida para fazer predicoes
     - Captura relacoes nao-lineares facilmente
   
   [-] Desvantagens:
     - Pode sofrer overfitting com arvores muito profundas
     - Sensivel a pequenas variacoes nos dados de treino
     - Pode criar regioes de decisao muito complexas

3. KNN (K-NEAREST NEIGHBORS):
   [+] Vantagens:
     - Simples e intuitivo
     - Nao faz suposicoes sobre distribuicao dos dados
     - Adapta-se bem a fronteiras de decisao irregulares
     - Performance ajustavel pelo parametro k
   
   [-] Desvantagens:
     - Computacionalmente custoso em grandes datasets
     - Sensivel a escala das features (requer normalizacao)
     - Performance depende da escolha de k
     - Armazena todo o conjunto de treino

4. INTERPRETACAO DAS MATRIZES DE CONFUSAO:
   As matrizes mostram que ambos os modelos tiveram poucos erros de
   classificacao. A maioria dos erros ocorreu na distincao entre as
   especies versicolor e virginica, que sao naturalmente mais similares.

5. ESCOLHA DO MODELO:
   - Para INTERPRETABILIDADE: Arvore de Decisao
   - Para DATASETS PEQUENOS: KNN pode ser mais eficiente
   - Para PRODUCAO: Considerar ensemble (Random Forest) ou validacao cruzada

6. METRICAS UTILIZADAS:
   - ACURACIA: Proporcao de predicoes corretas (VP + VN) / Total
   - PRECISAO: Proporcao de positivos preditos corretos VP / (VP + FP)
   - RECALL: Proporcao de positivos reais identificados VP / (VP + FN)
   - F1-SCORE: Media harmonica entre precisao e recall
     
     Onde: VP = Verdadeiros Positivos, VN = Verdadeiros Negativos
           FP = Falsos Positivos, FN = Falsos Negativos

7. OBSERVACOES SOBRE O PARAMETRO K:
   A analise mostrou que diferentes valores de k produzem diferentes
   acuracias. Valores muito baixos (k=1) podem causar overfitting,
   enquanto valores muito altos podem suavizar demais as fronteiras.
   O valor otimo geralmente esta entre 3 e 10 para este dataset.

8. PROXIMOS PASSOS SUGERIDOS:
   - Testar em outros datasets (Wine, Breast Cancer, Digits)
   - Aplicar validacao cruzada (cross-validation)
   - Ajustar hiperparametros com GridSearchCV
   - Experimentar ensemble methods (Random Forest, Gradient Boosting)
   - Analisar feature importance para entender quais atributos
     sao mais relevantes para a classificacao

9. LIMITACOES DESTE ESTUDO:
   - Dataset pequeno e bem comportado
   - Apenas uma divisao treino/teste (ideal: validacao cruzada)
   - Hiperparametros nao otimizados sistematicamente
   - Nao testado com dados desbalanceados
""")

print("\n" + "=" * 80)
print("FIM DA ANALISE")
print("=" * 80)
print("\n[OK] Codigo executado com sucesso!")
print("\nReferencias:")
print("   - Scikit-learn: https://scikit-learn.org")
print("   - UCI ML Repository: https://archive.ics.uci.edu/ml")
print("   - Fisher, R.A. (1936). Iris dataset")
print("="*80)