# Atividade Prática: Classificação Supervisionada
# Algoritmos: Árvore de Decisão e KNN
# Autor: Larissa Campos Cardoso
# Data: 28/10/2025

# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
import warnings
warnings.filterwarnings('ignore')

# Config Visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("ATIVIDADE PRÁTICA: CLASSIFICAÇÃO SUPERVISIONADA")
print("Algoritmos: Árvore de Decisão e KNN")
print("=" * 80)

# Carregamento dados e preparação

print("\n" + "=" * 80)
print("2. CARREGAMENTO E EXPLORAÇÃO DOS DADOS")
print("=" * 80)

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

df_iris = pd.DataFrame(X_iris, columns=iris.feature_names)
df_iris['target'] = y_iris
df_iris['species'] = df_iris['target'].map({
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
})

print("\n Dataset: IRIS")
print(f"   Dimensões: {X_iris.shape}")
print(f"   Classes: {iris.target_names}")
print(f"   Features: {iris.feature_names}")
print("\nPrimeiras linhas:")
print(df_iris.head())

print("\n Distribuição das classes:")
print(df_iris['species'].value_counts())

print("\n Estatísticas descritivas:")
print(df_iris.describe())

# Divisão de dados

print("\n" + "=" * 80)
print("3. DIVISÃO DOS DADOS")
print("=" * 80)

# treino (70%), teste (30%)

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, 
    test_size=0.3, 
    random_state=42, 
    stratify=y_iris
)

print(f"\n✓ Conjunto de treino: {X_train.shape[0]} amostras")
print(f"✓ Conjunto de teste: {X_test.shape[0]} amostras")
print(f"✓ Proporção: 70% treino / 30% teste")

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Dados normalizados (StandardScaler)")

# MODELO 1: ÁRVORE DE DECISÃO

print("\n" + "=" * 80)
print("4. MODELO 1: ÁRVORE DE DECISÃO")
print("=" * 80)

dt_model = DecisionTreeClassifier(
    max_depth=4, 
    random_state=42,
    criterion='gini'
)
dt_model.fit(X_train, y_train)

print("\n✓ Modelo treinado com sucesso!")
print(f"  Profundidade da árvore: {dt_model.get_depth()}")
print(f"  Número de folhas: {dt_model.get_n_leaves()}")

# Predições
y_pred_dt = dt_model.predict(X_test)

# Métricas
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt, average='weighted')
dt_recall = recall_score(y_test, y_pred_dt, average='weighted')
dt_f1 = f1_score(y_test, y_pred_dt, average='weighted')

print("\n MÉTRICAS - ÁRVORE DE DECISÃO:")
print(f"   Acurácia:  {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")
print(f"   Precisão:  {dt_precision:.4f}")
print(f"   Recall:    {dt_recall:.4f}")
print(f"   F1-Score:  {dt_f1:.4f}")

# Matriz de Confusão
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("\n Matriz de Confusão:")
print(cm_dt)

# Relatório 
print("\n Relatório de Classificação:")
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))

# MODELO 2: KNN 

print("\n" + "=" * 80)
print("5. MODELO 2: KNN (K-NEAREST NEIGHBORS)")
print("=" * 80)

# k=5
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train_scaled, y_train)

print("\n✓ Modelo KNN treinado com sucesso!")
print(f"  Número de vizinhos (k): {knn_model.n_neighbors}")
print(f"  Métrica de distância: {knn_model.metric}")

# Predições
y_pred_knn = knn_model.predict(X_test_scaled)

# Métricas
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn, average='weighted')
knn_recall = recall_score(y_test, y_pred_knn, average='weighted')
knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')

print("\n MÉTRICAS - KNN:")
print(f"   Acurácia:  {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
print(f"   Precisão:  {knn_precision:.4f}")
print(f"   Recall:    {knn_recall:.4f}")
print(f"   F1-Score:  {knn_f1:.4f}")

# Matriz de Confusão
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("\n📋 Matriz de Confusão:")
print(cm_knn)

# Relatório
print("\n Relatório de Classificação:")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

# Visualizar

print("\n" + "=" * 80)
print("6. VISUALIZAÇÕES")
print("=" * 80)

# Cria figura 
fig = plt.figure(figsize=(16, 10))

# Matriz de Confusão - Árvore de Decisão
plt.subplot(2, 3, 1)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Matriz de Confusão\nÁrvore de Decisão', fontsize=12, fontweight='bold')
plt.ylabel('Classe Real')
plt.xlabel('Classe Predita')

# Matriz de Confusão - KNN
plt.subplot(2, 3, 2)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Matriz de Confusão\nKNN', fontsize=12, fontweight='bold')
plt.ylabel('Classe Real')
plt.xlabel('Classe Predita')

# Comparação de Métricas
plt.subplot(2, 3, 3)
metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
dt_scores = [dt_accuracy, dt_precision, dt_recall, dt_f1]
knn_scores = [knn_accuracy, knn_precision, knn_recall, knn_f1]

x = np.arange(len(metrics))
width = 0.35

bars1 = plt.bar(x - width/2, dt_scores, width, label='Árvore de Decisão', color='#3498db')
bars2 = plt.bar(x + width/2, knn_scores, width, label='KNN', color='#2ecc71')

plt.ylabel('Score')
plt.title('Comparação de Métricas', fontsize=12, fontweight='bold')
plt.xticks(x, metrics, rotation=45)
plt.ylim([0.9, 1.01])
plt.legend()
plt.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

plt.subplot(2, 3, (4, 6))
plot_tree(dt_model, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=8)
plt.title('Estrutura da Árvore de Decisão', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('comparacao_modelos.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráficos gerados e salvos como 'comparacao_modelos.png'")
plt.show()

# 7. Análise KNN

print("\n" + "=" * 80)
print("7. ANÁLISE DE DIFERENTES VALORES DE K")
print("=" * 80)

k_values = range(1, 21)
k_scores = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    score = knn_temp.score(X_test_scaled, y_test)
    k_scores.append(score)

# Plotar
plt.figure(figsize=(10, 6))
plt.plot(k_values, k_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('Número de Vizinhos (k)', fontsize=12)
plt.ylabel('Acurácia', fontsize=12)
plt.title('Acurácia do KNN vs Valor de K', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)

# Marcar o melhor k
best_k = k_values[np.argmax(k_scores)]
best_score = max(k_scores)
plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Melhor k={best_k}')
plt.legend()

plt.tight_layout()
plt.savefig('knn_k_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Melhor valor de k: {best_k} (Acurácia: {best_score:.4f})")
plt.show()

# Análise final

print("\n" + "=" * 80)
print("8. ANÁLISE COMPARATIVA FINAL")
print("=" * 80)

print("\n" + "="*80)
print("RESUMO COMPARATIVO DOS MODELOS")
print("="*80)

comparison_df = pd.DataFrame({
    'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score'],
    'Árvore de Decisão': [dt_accuracy, dt_precision, dt_recall, dt_f1],
    'KNN (k=5)': [knn_accuracy, knn_precision, knn_recall, knn_f1]
})

print("\n", comparison_df.to_string(index=False))

# melhor modelo
if dt_f1 > knn_f1:
    melhor = "Árvore de Decisão"
    diferenca = dt_f1 - knn_f1
else:
    melhor = "KNN"
    diferenca = knn_f1 - dt_f1

print(f"\n🏆 MELHOR MODELO: {melhor}")
print(f"   Diferença de F1-Score: {diferenca:.4f}")

# Concluindo

print("\n" + "=" * 80)
print("9. CONCLUSÕES E INTERPRETAÇÕES")
print("=" * 80)

print("""
 ANÁLISE FINAL:

1. DESEMPENHO GERAL:
   Ambos os modelos apresentaram excelente desempenho no dataset Iris,
   com acurácias superiores a 95%. Isso se deve à natureza bem separável
   das classes neste dataset clássico.

2. ÁRVORE DE DECISÃO:
   ✓ Vantagens:
     - Interpretável e visual (podemos ver a lógica de decisão)
     - Não requer normalização dos dados
     - Rápida para fazer predições
     - Captura relações não-lineares facilmente
   
   ✗ Desvantagens:
     - Pode sofrer overfitting com árvores muito profundas
     - Sensível a pequenas variações nos dados de treino
     - Pode criar regiões de decisão muito complexas

3. KNN (K-NEAREST NEIGHBORS):
   ✓ Vantagens:
     - Simples e intuitivo
     - Não faz suposições sobre distribuição dos dados
     - Adapta-se bem a fronteiras de decisão irregulares
     - Performance ajustável pelo parâmetro k
   
   ✗ Desvantagens:
     - Computacionalmente custoso em grandes datasets
     - Sensível à escala das features (requer normalização)
     - Performance depende da escolha de k
     - Armazena todo o conjunto de treino

4. INTERPRETAÇÃO DAS MATRIZES DE CONFUSÃO:
   As matrizes mostram que ambos os modelos tiveram poucos erros de
   classificação. A maioria dos erros ocorreu na distinção entre as
   espécies versicolor e virginica, que são naturalmente mais similares.

5. ESCOLHA DO MODELO:
   - Para INTERPRETABILIDADE: Árvore de Decisão
   - Para DATASETS PEQUENOS: KNN pode ser mais eficiente
   - Para PRODUÇÃO: Considerar ensemble (Random Forest) ou validação cruzada

6. MÉTRICAS UTILIZADAS:
   - ACURÁCIA: Proporção de predições corretas
   - PRECISÃO: Proporção de positivos preditos que são realmente positivos
   - RECALL: Proporção de positivos reais que foram identificados
   - F1-SCORE: Média harmônica entre precisão e recall (mais robusta)

7. PRÓXIMOS PASSOS:
   - Testar em outros datasets (Wine, Breast Cancer)
   - Aplicar validação cruzada (cross-validation)
   - Ajustar hiperparâmetros (GridSearchCV)
   - Experimentar ensemble methods (Random Forest, Gradient Boosting)
   - Analisar curvas ROC e AUC para classificação multiclasse
""")

print("\n" + "=" * 80)
print("FIM DA ANÁLISE")
print("=" * 80)
print("\n✓ Notebook executado com sucesso!")
print("✓ Gráficos salvos: 'comparacao_modelos.png' e 'knn_k_analysis.png'")
print("\n Referências:")
print("   - Scikit-learn Documentation: https://scikit-learn.org")
print("   - UCI ML Repository: https://archive.ics.uci.edu/ml")
print("="*80)
