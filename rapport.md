# Rapport de Mini-Projet

**Module :** Fouille de Données 1  
**Intitulé :** Développement d'une Interface de Fouille de Données  
**Établissement :** Faculté d'Informatique  
**Année universitaire :** 2025–2026

---

## Résumé

Ce rapport présente la conception et le développement d'une interface interactive de fouille de données, réalisée dans le cadre du Mini-Projet FD1. L'application, développée en Python avec le framework Streamlit, automatise le pipeline complet de la fouille de données : du chargement et prétraitement des données jusqu'à l'évaluation de modèles d'apprentissage supervisé, en passant par le clustering. L'interface a été pensée pour être ergonomique, modulaire et réutilisable pour de futures analyses.

---

## 1. Introduction

La fouille de données (*data mining*) est une discipline centrale de l'intelligence artificielle et de l'analyse de données. Elle repose sur un pipeline structuré allant de la collecte et du nettoyage des données à l'extraction de connaissances utiles grâce à des algorithmes d'apprentissage automatique.

L'objectif de ce projet est de concevoir une interface graphique permettant à un utilisateur, même non expert, de réaliser l'intégralité de ce pipeline de manière visuelle et interactive. L'outil développé couvre trois grandes étapes : le prétraitement, le clustering (apprentissage non supervisé) et la classification (apprentissage supervisé).

---

## 2. Technologies et environnement de développement

### 2.1 Langage et framework

L'application est développée en **Python 3.10+** et utilise **Streamlit** comme framework d'interface web. Ce choix est motivé par :

- Sa simplicité de déploiement (une seule commande pour lancer l'application)
- Son intégration native avec les bibliothèques de data science Python
- Sa capacité à générer des interfaces réactives sans HTML/CSS avancé

### 2.2 Bibliothèques utilisées

| Bibliothèque | Rôle |
|---|---|
| `pandas` | Manipulation et analyse des données tabulaires |
| `numpy` | Calcul numérique et manipulation de tableaux |
| `matplotlib` / `seaborn` | Visualisation des données |
| `scikit-learn` | Algorithmes de clustering et de classification |
| `scipy` | Calculs statistiques, dendrogrammes hiérarchiques |

---

## 3. Architecture de l'application

L'application est structurée en un fichier principal `app.py` organisé en sections logiques correspondant aux trois volets du projet. La navigation se fait via un système d'onglets (*tabs*) Streamlit.

```
app.py
├── Configuration de la page et CSS
├── Sidebar — Chargement du fichier CSV
├── Onglet 1 — Prétraitement
│   ├── Vue d'ensemble et exploration
│   ├── Nettoyage et normalisation
│   └── Visualisations (Boxplot, Scatter Plot, Histogramme)
├── Onglet 2 — Clustering
│   ├── K-Means (avec courbe d'Elbow)
│   ├── K-Medoids (implémentation manuelle)
│   ├── DBSCAN, AGNES, DIANA, DENCLUE
│   └── Visualisation PCA 2D
└── Onglet 3 — Classification supervisée
    ├── Sélection de la variable cible et des features
    ├── Partitionnement train/test
    ├── KNN, Decision Tree, Random Forest, SVM, Régression Logistique
    └── Matrice de confusion, métriques, rapport de classification
```

---

## 4. Volet 1 — Prétraitement

### 4.1 Chargement des données

L'utilisateur charge un fichier CSV depuis la barre latérale. L'application affiche immédiatement le nombre de lignes, de colonnes et de valeurs manquantes.

### 4.2 Exploration

Un premier aperçu du dataset (10 premières lignes) est affiché, accompagné d'un tableau récapitulatif des types de données et du pourcentage de valeurs manquantes par colonne. Les statistiques descriptives (min, max, moyenne, écart-type, quartiles) sont accessibles via un menu dépliable.

### 4.3 Nettoyage

La gestion des valeurs manquantes est automatisée selon la stratégie suivante :

- **Colonnes numériques** : remplacement par la **moyenne** de la colonne
- **Colonnes catégorielles** : remplacement par le **mode** (valeur la plus fréquente)

### 4.4 Normalisation

Deux méthodes de mise à l'échelle sont proposées :

**Min-Max Scaler** : transforme les valeurs dans l'intervalle [0, 1]

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Standard Scaler (Z-score)** : centre et réduit les données

$$x' = \frac{x - \mu}{\sigma}$$

### 4.5 Visualisation

Trois types de graphiques sont disponibles de façon indépendante :

- **Boxplot** : détection des outliers sur les colonnes sélectionnées, avec comparaison avant/après normalisation
- **Scatter Plot** : nuage de points pour visualiser la relation entre deux variables
- **Histogramme** : distribution d'une variable avec choix du nombre de bins

---

## 5. Volet 2 — Clustering

### 5.1 Algorithmes implémentés

#### K-Means

L'algorithme K-Means partitionne les données en *k* clusters en minimisant l'inertie intra-cluster. La sélection automatique du *k* optimal est réalisée grâce à la **courbe d'Elbow** (inertie en fonction de *k*) et au **score de Silhouette**. Le *k* maximisant le score de Silhouette est suggéré, mais l'utilisateur peut le modifier.

#### K-Medoids

Contrairement à K-Means qui utilise des centroïdes (moyennes), K-Medoids choisit des points réels du dataset comme représentants des clusters (*medoïdes*). L'algorithme PAM (Partitioning Around Medoids) est implémenté manuellement. Il est plus robuste aux outliers et supporte des métriques de distance alternatives (Manhattan, Cosine).

#### DBSCAN

Algorithme de clustering basé sur la densité. Il identifie des clusters de forme arbitraire et détecte automatiquement les points de bruit. Deux paramètres sont à régler : *epsilon* (rayon de voisinage) et *min_samples* (nombre minimum de points dans un voisinage).

#### AGNES (Agglomerative Nesting)

Algorithme de clustering hiérarchique ascendant. Commence avec chaque point dans son propre cluster et fusionne progressivement. Un dendrogramme est généré pour visualiser la hiérarchie. Quatre méthodes de linkage sont proposées : *ward*, *complete*, *average*, *single*.

#### DIANA (Divisive Analysis)

Approche inverse de AGNES : commence avec un seul cluster contenant tous les points et le divise récursivement. Implémenté via la méthode de linkage *complete* de scipy.

#### DENCLUE

Algorithme basé sur l'estimation de densité par noyau gaussien. Les clusters sont formés autour des maxima locaux de densité.

### 5.2 Évaluation

Deux métriques d'évaluation interne sont calculées pour tous les algorithmes :

- **Score de Silhouette** ∈ [-1, 1] : mesure la cohésion intra-cluster et la séparation inter-cluster. Une valeur proche de 1 indique des clusters bien définis.
- **Indice de Davies-Bouldin** : ratio de la similarité intra-cluster sur la dissimilarité inter-cluster. Une valeur faible indique une meilleure partition.

### 5.3 Visualisation

Les résultats sont projetés en 2D via une **Analyse en Composantes Principales (PCA)**, permettant de visualiser les clusters quel que soit le nombre de dimensions initial. Le pourcentage de variance expliquée par chaque composante est affiché.

---

## 6. Volet 3 — Classification Supervisée

### 6.1 Préparation des données

L'utilisateur sélectionne :

- La **variable cible** (colonne à prédire, encodée automatiquement si catégorielle via `LabelEncoder`)
- Les **features** (variables d'entrée)
- Le **ratio train/test** (entre 10 % et 40 %) avec option de stratification pour respecter la distribution des classes

### 6.2 Modèles de classification

Cinq algorithmes sont intégrés avec paramétrage des hyperparamètres :

**K-Nearest Neighbors (KNN)** : classification par vote des *k* voisins les plus proches. Simple et efficace pour les datasets de taille modérée.

**Decision Tree** : arbre de décision construit par partitionnement récursif selon le critère de Gini ou d'entropie. Facilement interprétable.

**Random Forest** : ensemble d'arbres de décision entraînés sur des sous-ensembles aléatoires des données (*bagging*). Plus robuste et moins sujet au surapprentissage.

**Support Vector Machine (SVM)** : cherche l'hyperplan de séparation maximisant la marge entre les classes. Noyaux disponibles : RBF, linéaire, polynomial.

**Logistic Regression** : modèle linéaire probabiliste, référence pour la classification binaire et multiclasse.

### 6.3 Évaluation

Les performances sont évaluées sur le jeu de test via :

**Accuracy** : proportion de prédictions correctes
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision** : parmi les prédictions positives, quelle proportion est correcte
$$Precision = \frac{TP}{TP + FP}$$

**Recall (Sensibilité)** : parmi les vrais positifs, quelle proportion est correctement identifiée
$$Recall = \frac{TP}{TP + FN}$$

**F1-Score** : moyenne harmonique de la Precision et du Recall
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

La **matrice de confusion** est affichée sous forme de heatmap, permettant d'identifier les classes les plus souvent confondues.

Pour les modèles basés sur des arbres (Decision Tree, Random Forest), un graphique d'**importance des features** est généré, permettant d'identifier les variables les plus discriminantes.

---

## 7. Interface et expérience utilisateur

L'interface suit un thème sombre moderne avec une palette cohérente (violet #6C63FF, vert menthe #00D4AA). La typographie combine *Space Mono* pour les titres et métriques et *DM Sans* pour le corps de texte.

Le flux de travail est guidé : l'utilisateur doit obligatoirement prétraiter les données avant d'accéder aux onglets de clustering et de classification. Des messages d'information clairs indiquent les étapes à suivre.

---

## 8. Conclusion

L'interface développée répond à l'ensemble des exigences du Mini-Projet FD1. Elle offre un pipeline complet et réutilisable couvrant les trois volets demandés. Les choix techniques (Python, Streamlit, scikit-learn) garantissent la maintenabilité et l'extensibilité de l'outil. Des améliorations futures pourraient inclure le support de formats de données supplémentaires (Excel, JSON), l'ajout de techniques de réduction de dimensionnalité avancées (t-SNE, UMAP) ou l'export automatique des rapports d'évaluation.

---

*Rapport rédigé dans le cadre du Mini-Projet FD1 — Fouille de Données 1*  
*Faculté d'Informatique — Année universitaire 2025–2026*
