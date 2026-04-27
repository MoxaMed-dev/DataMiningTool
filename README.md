# ⚗️ DataMining Suite — Mini-Projet FD1

Interface interactive de fouille de données développée avec **Streamlit**.  
Pipeline complet : Prétraitement · Clustering · Classification supervisée.

---

## 📁 Arborescence du projet

```
mini-projet-fd1/
│
├── app.py              # Application principale Streamlit
├── requirements.txt    # Dépendances Python
├── README.md           # Ce fichier
├── rapport.md          # Rapport du projet (à convertir en PDF)
│
└── data/               # (optionnel) Datasets de test
    └── exemple.csv
```

---

## ⚙️ Prérequis

- **Python** 3.9 ou supérieur
- **pip** (gestionnaire de paquets Python)

---

## 🚀 Installation & Lancement

### 1. Cloner ou télécharger le projet

```bash
git clone <url-du-repo>
cd mini-projet-fd1
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Créer l'environnement
python -m venv venv

# Activer — Windows
venv\Scripts\activate

# Activer — macOS / Linux
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse :  
**http://localhost:8501**

---

## 📦 Dépendances (`requirements.txt`)

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

---

## 🧭 Guide d'utilisation

### Étape 1 — Charger les données

Dans la **barre latérale gauche**, cliquez sur **"Browse files"** et sélectionnez un fichier `.csv`.  
Les informations (nombre de lignes, colonnes) s'affichent immédiatement.

---

### 📋 Onglet 1 — Prétraitement

| Fonctionnalité | Description                                                                   |
| -------------- | ----------------------------------------------------------------------------- |
| Vue d'ensemble | Aperçu du dataset (10 premières lignes, métriques clés)                       |
| Exploration    | Statistiques descriptives, types et valeurs manquantes                        |
| Nettoyage      | Imputation automatique (moyenne pour numérique, mode pour catégoriel)         |
| Normalisation  | Min-Max Scaler ou Standard Scaler (Z-score)                                   |
| Boxplot        | Sélectionnez les colonnes à afficher, visualisation avant/après normalisation |
| Scatter Plot   | Choisissez les axes X et Y                                                    |
| Histogramme    | Distribution d'une variable avec choix du nombre de bins                      |
| Export         | Téléchargement du dataset prétraité en `.csv`                                 |

> ⚠️ Le prétraitement doit être appliqué avant d'utiliser les onglets Clustering et Classification.

---

### 🔵 Onglet 2 — Clustering

| Algorithme    | Paramètres principaux                                      |
| ------------- | ---------------------------------------------------------- |
| **K-Means**   | k automatique (Elbow + Silhouette), ajustable manuellement |
| **K-Medoids** | k, métrique de distance, random state                      |
| **DBSCAN**    | Epsilon, min_samples, métrique                             |
| **AGNES**     | k, méthode de linkage (ward/complete/average/single)       |
| **DIANA**     | k (clustering divisif hiérarchique)                        |
| **DENCLUE**   | Sigma, Xi (seuil de densité)                               |

Tous les résultats sont visualisés en **2D via PCA** avec le score de Silhouette et l'indice Davies-Bouldin.

---

### 🤖 Onglet 3 — Classification Supervisée

1. **Sélectionner la variable cible** (colonne à prédire)
2. **Choisir les features** d'entrée
3. **Configurer la partition** train/test (10–40 %, stratification optionnelle)
4. **Choisir un algorithme** et régler ses hyperparamètres :

| Modèle              | Hyperparamètres                        |
| ------------------- | -------------------------------------- |
| KNN                 | k, métrique                            |
| Decision Tree       | profondeur max, critère (gini/entropy) |
| Random Forest       | nb d'arbres, profondeur max            |
| SVM                 | noyau (rbf/linear/poly), C             |
| Logistic Regression | C, solver                              |

5. Cliquer **"Entraîner & Évaluer"** pour obtenir :
   - Accuracy, Precision, Recall, F1-Score
   - Matrice de Confusion
   - Rapport de classification complet
   - Importance des features (Random Forest / Decision Tree)

---

## 👤 Auteur

BOUDA Mohamed

2026 © DATA MINING TOOL
