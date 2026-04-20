Deep Pricing — Estimating S&P 500 Options Pricing using Deep Learning

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

Une solution est considérée réussie si la MAE du MLP est inférieure à la MAE baseline de Black-Scholes (1,48 $). 
Le modèle atteint finalement une MAE de 0,63$, soit une amélioration de 57%.

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

•	Numériques continus : UNDERLYING_LAST (prix spot, ~350–480 $), STRIKE (prix d’exercice, ~300–550 $), IV (volatilité implicite, ~0,10–1,50), Option_Price (variable cible, ~0,01–200 $) BID / ASK, Delta, Gamma etc. Au total 33 colonnes.

•	Numérique entier : DTE (Days to Expiry, de 1 à ~500 jours).

•	Binaire (cas particulier de catégorielle) : Is_Call (1 = Call, 0 = Put), encodée directement en entier, sans one-hot encoding nécessaire grâce à sa nature binaire.

NB : La variable QUOTE_DATE (date de cotation) est présente dans le dataset brut mais n’est pas utilisée comme feature, elle sert uniquement à la sélection de la période 2020–2022. Le dataset ne modélise donc pas de dépendance temporelle explicite ; DTE capte indirectement la dimension temporelle propre à chaque contrat.

---

## Data Distribution

📌 **Instructions:**  
Describe important distribution characteristics.

Examples:

- Class balance or imbalance
- Skewed numerical variables
- Range of key features

✏️ **Describe the data distribution here**

La distribution des prix d'options, qui constitue notre variable cible, est fortement asymétrique à droite. La grande majorité des contrats correspondent à des options Out-of-the-Money (OTM) dont le prix est faible, souvent inférieur à quelques dollars, tandis qu'une minorité d'options deep In-the-Money (ITM) atteignent des prix de plusieurs centaines de dollars. Cette asymétrie est une caractéristique fondamentale des marchés d'options et justifie l'utilisation de la MAE plutôt que de la RMSE comme métrique principale, car cette dernière serait trop influencée par ces valeurs extrêmes.

Concernant les variables d'entrée, UNDERLYING_LAST et STRIKE évoluent toutes deux dans une plage similaire, approximativement entre 300 $ et 550 $ sur la période 2020–2022, ce qui reflète les niveaux de l'ETF SPY durant cette période. La variable IV (volatilité implicite) présente également une distribution asymétrique : les valeurs sont concentrées autour de 0,20–0,40 en période normale, mais atteignent des niveaux bien supérieurs lors du krach COVID de mars 2020, créant une queue de distribution importante vers la droite. DTE varie de 1 à environ 500 jours, avec une concentration marquée sur les échéances courtes (moins de 60 jours), qui sont historiquement les plus échangées sur le marché des options.

Sur la question de l'équilibre entre Calls et Puts, le dataset est relativement équilibré après restructuration en format long, ce qui évite tout biais lié à une sur-représentation d'un type d'option. Enfin, parmi les corrélations notables, IV et STRIKE sont négativement corrélés (−0,52), ce qui traduit le phénomène bien connu du smile de volatilité : les options à strike bas tendent à avoir une volatilité implicite plus élevée. DTE et IV sont quant à eux positivement corrélés avec Option_Price, ce qui est cohérent avec la théorie financière.

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

Deux modèles ont été utilisés dans ce projet, avec des rôles distincts.

Le premier est le modèle de Black-Scholes-Merton (1973), qui joue le rôle de baseline. Il s'agit d'une formule analytique fermée qui calcule le prix théorique d'une option européenne à partir de cinq paramètres : le prix du sous-jacent, le prix d'exercice, le temps avant échéance, la volatilité implicite et le taux sans risque. Nous l'avons implémenté en Python avec scipy.stats.norm et appliqué séparément aux Calls et aux Puts via la formule BSM standard. Bien que Black-Scholes soit conçu pour des options européennes alors que les options SPY sont américaines, il reste la référence incontournable du marché et constitue donc un point de comparaison légitime et exigeant.

Le second est un MLP (Multilayer Perceptron) implémenté en PyTorch, qui constitue le modèle principal. Il s'agit d'un réseau de neurones fully connected composé de trois couches cachées (64 → 32 → 16 neurones) avec activation ReLU, et d'une couche de sortie à un neurone avec activation Softplus. Le MLP apprend directement depuis les données de marché, sans aucune hypothèse paramétrique sur la distribution des rendements ou la constance de la volatilité. C'est précisément ce qui lui permet de capturer des dynamiques de prix que Black-Scholes ne peut pas modéliser, notamment lors de régimes de volatilité extrêmes comme ceux observés en 2020 et 2022.

---

## Modeling Strategy

📌 **Instructions:**  
Explain:

- Why you selected these models
- Whether you used a baseline model
- If hyperparameter tuning was performed
- Whether cross-validation was used

✏️ **Explain your modeling strategy**

Nous avons choisi Black-Scholes comme modèle baseline car il constitue la référence standard en finance pour le pricing d'options. Son utilisation comme point de comparaison est naturelle : si notre modèle de deep learning ne parvient pas à le surpasser, cela remettrait en question l'intérêt de l'approche. Black-Scholes nous fournit donc un seuil minimal de performance à dépasser (MAE = 1,48 $).

Le MLP (Multilayer Perceptron) a été sélectionné comme modèle principal pour plusieurs raisons. D'abord, le pricing d'options est un problème intrinsèquement non-linéaire : la relation entre les Greeks (Delta, Gamma, Vega) et le prix d'une option ne peut pas être capturée par un modèle linéaire. Ensuite, le MLP est particulièrement adapté aux données tabulaires structurées comme les nôtres, contrairement aux CNN ou LSTM qui sont pensés pour des données spatiales ou séquentielles. Enfin, le MLP est largement utilisé dans la littérature académique récente sur le deep learning appliqué au pricing d'options, ce qui rend nos résultats comparables aux travaux existants.

Concernant le tuning des hyperparamètres, nous avons fixé l'architecture (64 → 32 → 16 neurones), le taux d'apprentissage (0,001), la taille de batch (512) et le nombre d'époques (30) sur la base de valeurs standards pour ce type de problème, sans effectuer de recherche systématique par grid search ou random search. Une optimisation plus rigoureuse des hyperparamètres constituerait une amélioration future pertinente.

Enfin, aucune cross-validation n'a été mise en place. Nous avons opté pour un split aléatoire simple 80/20. Il convient de noter qu'un split temporel, par exemple entraîner sur 2020–2021 et tester sur 2022, aurait été méthodologiquement plus robuste pour des données financières, car il éviterait tout risque de data leakage entre des contrats du même jour présents à la fois dans le train et dans le test.

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

Deux métriques ont été utilisées dans ce projet, chacune jouant un rôle distinct.

La MAE (Mean Absolute Error) est la métrique principale d'évaluation. Elle mesure l'erreur moyenne en valeur absolue entre le prix prédit et le prix de marché réel, exprimée directement en dollars. C'est la métrique la plus appropriée dans notre contexte pour deux raisons : d'une part, elle est immédiatement interprétable, une MAE de 0,63 $ signifie concrètement que le modèle se trompe en moyenne de 63 centimes par contrat, et d'autre part, elle permet une comparaison directe et intuitive avec le baseline Black-Scholes (MAE = 1,48 $). La MAE est également moins sensible aux valeurs extrêmes que la MSE, ce qui la rend plus représentative de la performance moyenne du modèle sur l'ensemble des contrats.

La MSE (Mean Squared Error) est utilisée comme fonction de perte pendant l'entraînement du MLP, mais pas comme métrique d'évaluation finale. Le choix de la MSE comme loss function est délibéré : en pénalisant les grandes erreurs de manière quadratique, elle pousse l'optimiseur à réduire en priorité les erreurs importantes, notamment sur les options deep In-the-Money dont le prix peut atteindre plusieurs centaines de dollars. Ce comportement est souhaitable en finance, où une erreur de pricing importante sur un seul contrat peut avoir des conséquences significatives.

En résumé, la combinaison MSE (loss d'entraînement) et MAE (métrique d'évaluation) est un choix cohérent : on entraîne le modèle à minimiser les grandes erreurs, et on l'évalue sur sa précision moyenne exprimée dans une unité directement compréhensible.

---

# 8. Project Structure

📌 **Instructions:**  
Explain how the repository is organized.

If you added additional folders, explain them.

Le projet est organisé de manière simple autour de deux fichiers principaux, sans structure de dossiers complexe, ce qui reflète la nature exploratoire et académique du travail réalisé sur Google Colab.

Le fichier central est le notebook IA_in_Finance_Tharreau_Jenner_Morton.ipynb, qui contient l'intégralité de la pipeline : chargement et nettoyage des données, analyse exploratoire, implémentation du baseline Black-Scholes, construction et entraînement du MLP, et évaluation comparative des deux modèles. Tout le code est regroupé dans ce fichier unique, organisé en sections séquentielles correspondant aux grandes étapes du projet.

Le fichier DeepPricing_v3.pptx est la présentation associée au projet. Elle résume en 10 slides la problématique, le dataset, la méthodologie et les résultats obtenus.

Le dataset brut spy_2020_2022.csv n'est pas versionné dans le dépôt. Il est stocké directement sur Google Drive et chargé dans le notebook via un montage Drive (drive.mount). Ce choix s'explique par la taille du fichier, qui dépasse les limites raisonnables pour un versionnement Git classique.

---

# 9. Installation

📌 **Instructions:**  
Explain how to install project dependencies.

Le projet tourne entièrement sur Google Colab, ce qui signifie que la grande majorité des dépendances sont déjà préinstallées dans l'environnement Colab et ne nécessitent aucune installation manuelle. Il suffit d'ouvrir le notebook et d'exécuter les cellules dans l'ordre.

La seule étape préalable nécessaire est de monter Google Drive pour accéder au dataset, ce qui est géré directement dans la première cellule du notebook :
pythonfrom google.colab import drive 
drive.mount('/content/drive')

Le dataset spy_2020_2022.csv doit être placé dans le dossier suivant sur votre Google Drive :
/content/drive/MyDrive/AI_Finance_Project/spy_2020_2022.csv

Les bibliothèques utilisées dans le projet sont les suivantes, toutes disponibles nativement sur Colab :
bashpip install pandas numpy matplotlib seaborn scikit-learn torch scipy

Pour toute exécution en local, en dehors de Colab, il est recommandé de créer un environnement virtuel avant d'installer les dépendances :
bashpython -m venv venv

source venv/bin/activate       # Sur Mac/Linux
venv\Scripts\activate          # Sur Windows

pip install pandas numpy matplotlib seaborn scikit-learn torch scipy


Example:

```bash
pip install -r requirements.txt
