<img width="470" height="150" alt="image" src="https://github.com/user-attachments/assets/4b93f545-de37-414a-807c-0bfc24d94f54" />Deep Pricing — Estimating S&P 500 Options Pricing using Deep Learning

---

# 1. Project Information

Fill in the following information.

- **Project Title:** Deep Pricing — Estimating S&P 500 Options Pricing using Deep Learning
- **Group Name:** DeepPricing Team
- **Group Members:**  
  - Student 1 – Lukas Tharreau
  - Student 2 – Jenner Bastien
  - Student 3 – Alexandre Morton

- **Course Name:** AI In Finance
- **Instructor:** Nicolas De Roux & Mohamed EL FAKIR
- **Submission Date:** 20/04/2026

---

# 2. Project Description

📌 **Instructions:**  
Write a short paragraph explaining the **problem addressed by the project**.

Include:

- The context of the problem
- Why the problem is interesting or important
- Who might benefit from solving it

✏️ **Write your description below:**

Le pricing des options financières est une problématique centrale en finance quantitative. Le modèle de Black-Scholes (1973) constitue la référence historique, mais repose sur des hypothèses strictes : volatilité constante, rendements log-normaux, et conception pour les seules options européennes. Ces hypothèses sont régulièrement violées en pratique, notamment lors de crises financières majeures comme le krach COVID de mars 2020 ou le bear market inflationniste de 2022.

Ce projet explore si un réseau de neurones profond (MLP) entraîné sur des données de marché réelles peut apprendre à pricer des options de manière plus précise que la formule analytique classique, sans imposer aucune hypothèse paramétrique. Ce sujet intéresse aussi bien les banques d'investissement et les fonds quantitatifs que les équipes de risk management, qui cherchent des outils de pricing plus robustes face aux régimes de volatilité extrêmes.


---

# 3. Project Goal

📌 **Instructions:**  
Clearly explain **what the project aims to achieve**.

Your answer should describe:

- What the system predicts, classifies, or analyzes
- What a successful solution looks like

Example goals:

- Predict housing prices from property features
- Classify medical images into disease categories
- Detect sentiment in text reviews
- Forecast electricity consumption

✏️ **Write your project goal below:**

L'objectif du projet est de construire un modèle de deep learning (MLP PyTorch) capable de prédire le prix de marché d'une option SPY (ETF répliquant le S&P 500) à partir de cinq variables financières clés, en surpassant la précision du modèle Black-Scholes mesuré par la Mean Absolute Error (MAE).

Une solution est considérée réussie si la MAE du MLP est inférieure à la MAE baseline de Black-Scholes (1,48 $). Le modèle atteint finalement une MAE de 0,63 $, soit une amélioration de 57 %.

---

# 4. Task Definition

📌 **Instructions:**  
Define the **machine learning or data analysis task**.

Specify:

- **Task Type:** (classification, regression, clustering, etc.)
- **Input:** What data is used as input
- **Output:** What the model predicts
- **Evaluation Metric:** How performance will be measured

✏️ **Fill in the following:**

- **Task Type:** Régression supervisée
- **Input Variables:** UNDERLYING_LAST (prix du sous-jacent), STRIKE (prix d'exercice), DTE (jours avant échéance), IV (volatilité implicite), Is_Call (1=Call / 0=Put)
- **Target Variable:**  Option_Price — midpoint (Bid + Ask) / 2, en dollars
- **Evaluation Metric(s):** Primary Metric	MAE (Mean Absolute Error) — directement interprétable en dollars / Secondary Metric	MSE (Mean Squared Error) — utilisé comme loss function pendant l'entraînement

---

# 5. Dataset Description

📌 **Instructions:**  
Describe the dataset used in the project.

❗ Do **not simply name the dataset**. Instead explain its structure and contents.

---

## Dataset Overview

Provide general information about the dataset.

Fill in:

- **Number of samples:**  3,5M
- **Number of features:**  5
- **Target variable:**  Option_Price
- **Data source:** Kaggle — Historical SPY Options Data : https://drive.google.com/file/d/13z_aJ2pP5FkGF73ilMMCTS3ufYHMA80I/view?usp=drive_link

---

## Feature Description

📌 **Instructions:**  
List and describe the most important variables.

Example table:

| Feature | Description | Type |
|------|------|------|
| age | Age of individual | Numerical |
| income | Annual income | Numerical |
| gender | Gender category | Categorical |

✏️ **Insert your feature description table here**

| Feature | Description | Type |
|------|------|------|
| Spot Price | Prix actuel de l'actif sous-jacent (S&P 500) | Numerical |
| Strike Price | Prix d'exercice du contrat d'option | Numerical |
| Time to Maturity (DTE) | Temps restant (en jours) avant l'expiration de l'option | Numerical |
| Implied Volatility (IV) | Volatilité implicite attendue par le marché | Numerical |
| Bid | Prix d'achat le plus élevé proposé sur le marché | Numerical |
| Ask | Prix de vente le plus bas exigé sur le marché | Numerical |
| Volume | Nombre total de contrats échangés sur la période | Numerical |
| Is_Call | Variable binaire indiquant s'il s'agit d'un Call (1) ou d'un Put (0) | Categorical |
| Option_Price | Prix de l'option calculé (Moyenne entre le Bid et le Ask) - Variable cible | Numerical |

---

## Target Variable

📌 **Instructions:**  
Explain what the model is trying to predict.

Include:

- Variable name
- Meaning
- Possible values (if classification)

✏️ **Write your explanation here**

La variable cible est Option_Price, calculée comme le midpoint entre le Bid et l'Ask. Ce prix médian est préféré au dernier prix échangé car il réduit le bruit de microstructure de marché. Il prend des valeurs positives continues, généralement de 0,01 $ à plusieurs centaines de dollars pour les options deep ITM.

---

## Data Types

📌 **Instructions:**  
Describe the types of variables present in the dataset.

Examples:

- Numerical
- Categorical
- Ordinal
- Text
- Time-series

✏️ **Describe the column types here**

Le dataset contient exclusivement des variables numériques, aucune variable textuelle, ordinale ou de type time-series n’est utilisée directement comme feature dans le modèle :
●	Numériques continus : UNDERLYING_LAST (prix spot, ~350–480 $), STRIKE (prix d’exercice, ~300–550 $), IV (volatilité implicite, ~0,10–1,50), Option_Price (variable cible, ~0,01–200 $) BID / ASK, Delta, Gamma etc. Au total 33 colonnes.
●	Numérique entier : DTE (Days to Expiry, de 1 à ~500 jours).
●	Binaire (cas particulier de catégorielle) : Is_Call (1 = Call, 0 = Put), encodée directement en entier, sans one-hot encoding nécessaire grâce à sa nature binaire.

Note : La variable QUOTE_DATE (date de cotation) est présente dans le dataset brut mais n’est pas utilisée comme feature, elle sert uniquement à la sélection de la période 2020–2022. Le dataset ne modélise donc pas de dépendance temporelle explicite ; DTE capte indirectement la dimension temporelle propre à chaque contrat.

---

## Data Distribution

📌 **Instructions:**  
Describe important distribution characteristics.

Examples:

- Class balance or imbalance
- Skewed numerical variables
- Range of key features

✏️ **Describe the data distribution here**

La distribution des prix d'options est fortement asymétrique à droite (right-skewed) : la majorité des contrats sont des options bon marché Out-of-the-Money (OTM) à faible prix, avec une longue queue de distribution pour les options deep In-the-Money (ITM) très chères. Les corrélations clés observées : IV et STRIKE sont négativement corrélés (−0,52), tandis que DTE et IV sont positivement corrélés avec Option_Price.

---

## Data Quality

📌 **Instructions:**  
Mention any issues found in the dataset.

Examples:

- Missing values
- Outliers
- Imbalanced classes
- Duplicate entries

✏️ **Describe any data quality issues here**

Plusieurs problèmes de qualité ont été identifiés et traités :
•	Noms de colonnes malformés dans le CSV brut (espaces, crochets) → nettoyés par regex
•	Colonnes numériques lues comme strings par Pandas → converties avec pd.to_numeric(errors='coerce')
•	Contrats illiquides (volume = 0) → filtrés (bruit de prix stale)
•	Options 0-DTE (expirant le jour même) → exclues (comportement Gamma instable)
•	Valeurs manquantes sur IV, BID, ASK → supprimées par dropna()
•	Données initiales en format 'large' (Call + Put sur la même ligne) → restructurées en format 'long'

---

# 6. Data Preprocessing

📌 **Instructions:**  
Explain the preprocessing steps applied before modeling.

Examples:

- Handling missing values
- Removing duplicates
- Encoding categorical variables
- Normalizing or scaling features
- Feature engineering

For each step briefly explain **why it was necessary**.

✏️ **Describe your preprocessing steps here**

Les étapes de prétraitement suivantes ont été appliquées, dans l'ordre :
•	Nettoyage des noms de colonnes : suppression des espaces et crochets par expression régulière.
•	Conversion des types : forçage de toutes les colonnes de prix et volume en float (errors='coerce' pour transformer les valeurs non-parsables en NaN).
•	Restructuration Long Format : séparation des Calls et Puts (initialement sur la même ligne) en deux ensembles de lignes indépendantes, avec création de la variable binaire Is_Call.
•	Calcul de la variable cible : Option_Price = (BID + ASK) / 2.
•	Filtres de liquidité : exclusion des contrats avec volume = 0 ou DTE = 0.
•	Suppression des NaN résiduels sur les colonnes critiques (IV, BID, ASK).
•	Échantillonnage : tirage aléatoire stratifié de 200 000 contrats (random_state=42) pour contrainte mémoire Google Colab.
•	Normalisation des features : StandardScaler (mean=0, std=1) sur les 5 variables d'entrée, indispensable pour éviter les gradients explosifs entre des variables d'échelles très différentes (STRIKE ~400, DTE ~60, IV ~0.2).
•	Train/Test split : 80% entraînement / 20% test (split aléatoire).

---

# 7. Modeling Approach

📌 **Instructions:**  
Explain how you solved the problem.


---

## Chosen Models

List the models or algorithms used.

Examples:

- Linear Regression
- Logistic Regression
- Random Forest
- Gradient Boosting
- Neural Networks

✏️ **List and describe the models used**

---

## Modeling Strategy

📌 **Instructions:**  
Explain:

- Why you selected these models
- Whether you used a baseline model
- If hyperparameter tuning was performed
- Whether cross-validation was used

✏️ **Explain your modeling strategy**

---

## Evaluation Metrics

📌 **Instructions:**  
Specify the metrics used to evaluate model performance.

Examples:

- Accuracy
- Precision / Recall
- F1-score
- ROC-AUC
- Mean Absolute Error
- RMSE

Also explain **why these metrics are appropriate**.

✏️ **Describe your evaluation metrics**

---

# 8. Project Structure

📌 **Instructions:**  
Explain how the repository is organized.


If you added additional folders, explain them.

---

# 9. Installation

📌 **Instructions:**  
Explain how to install project dependencies.

Example:

```bash
pip install -r requirements.txt
