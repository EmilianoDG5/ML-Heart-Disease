# ❤️ Heart Disease Prediction using Machine Learning

Progetto di Machine Learning per la predizione delle malattie cardiache utilizzando algoritmi di classificazione supervisionata e tecniche di interpretabilità del modello.

## 📌 Descrizione

L'obiettivo del progetto è analizzare un dataset clinico relativo a pazienti affetti o meno da malattie cardiache e costruire modelli di Machine Learning in grado di prevedere la presenza di una patologia cardiaca.

Il progetto comprende:

- Pulizia e preparazione dei dati
- Analisi esplorativa del dataset (EDA)
- Addestramento di modelli di classificazione
- Valutazione delle performance
- Analisi dell'importanza delle feature
- Interpretabilità tramite SHAP

---

## 📂 Struttura del progetto

```text
ML-Heart-Disease/
│
├── heart.csv
├── clean.py
├── analisidati.py
├── classificatori.py
│
├── output_distribution.png
├── output_vs_age.png
├── output_vs_sex.png
├── output_vs_cp.png
│
├── confusion_matrix_Decision_Tree.png
├── confusion_matrix_Random_Forest.png
├── roc_curve_comparison.png
│
├── feature_importance_decision_tree.png
├── feature_importance_random_forest.png
│
├── shap_feature_importance_decision_tree.png
├── shap_feature_importance_random_forest.png
│
└── README.md
```

---

## 🧹 Data Cleaning

Il file `clean.py` esegue le operazioni di preprocessing sul dataset:

### Rimozione duplicati

- Individuazione delle righe duplicate
- Eliminazione dei duplicati presenti nel dataset

### Gestione valori mancanti

Per ogni colonna:

- Variabili numeriche → sostituzione con la mediana
- Variabili categoriche → sostituzione con la moda

### Analisi degli outlier

Viene utilizzato il metodo dell'Interquartile Range (IQR):

- Calcolo di Q1 e Q3
- Individuazione degli outlier
- Report del numero di valori anomali per ogni feature

---

## 📊 Analisi Esplorativa dei Dati

Il file `analisidati.py` genera diverse visualizzazioni statistiche:

### Distribuzione della variabile target

Visualizzazione della distribuzione dei pazienti:

- 0 = Nessuna malattia cardiaca
- 1 = Presenza di malattia cardiaca

### Analisi rispetto all'età

Istogramma della distribuzione dei casi in funzione dell'età.

### Analisi rispetto al sesso

Confronto tra pazienti maschi e femmine.

### Analisi rispetto al tipo di dolore toracico

Studio della relazione tra la variabile `cp` (Chest Pain Type) e la presenza della malattia.

---

## 🤖 Modelli di Machine Learning

Nel file `classificatori.py` vengono addestrati due algoritmi:

### Decision Tree

Classificatore basato su una struttura gerarchica di decisioni che suddivide i dati in base alle feature più informative.

### Random Forest

Modello ensemble composto da molteplici alberi decisionali che migliora la capacità di generalizzazione e riduce il rischio di overfitting.

---

## 📈 Metriche di valutazione

I modelli vengono confrontati utilizzando:

- Accuracy
- Precision
- Recall
- F1-Score
- Matrice di Confusione
- ROC Curve
- Area Under Curve (AUC)

---

## 🔍 Interpretabilità del modello

Per comprendere quali variabili influenzano maggiormente le predizioni vengono utilizzate due tecniche:

### Feature Importance

Calcolata direttamente dai modelli:

- Feature Importance - Decision Tree
- Feature Importance - Random Forest

### SHAP (SHapley Additive Explanations)

Tecnica avanzata che permette di interpretare il contributo delle singole feature alle predizioni del modello.

Output generati:

- SHAP Feature Importance - Decision Tree
- SHAP Feature Importance - Random Forest

---

## ⚙️ Tecnologie utilizzate

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- SHAP

---

## 📥 Installazione

Clonare il repository:

```bash
git clone https://github.com/EmilianoDG5/ML-Heart-Disease.git
cd ML-Heart-Disease
```

Creare un ambiente virtuale:

```bash
python -m venv venv
```

Attivarlo:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Installare le dipendenze:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

---

## ▶️ Esecuzione

### 1. Pulizia del dataset

```bash
python clean.py
```

### 2. Analisi esplorativa dei dati

```bash
python analisidati.py
```

### 3. Addestramento e valutazione dei modelli

```bash
python classificatori.py
```

---

## 📊 Dataset

Il dataset contiene informazioni cliniche dei pazienti, tra cui:

- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate Achieved
- Exercise Induced Angina
- ST Depression
- Altri parametri cardiologici

### Variabile target

| Valore | Significato |
|---------|------------|
| 0 | Nessuna malattia cardiaca |
| 1 | Presenza di malattia cardiaca |

---

## 🎯 Obiettivi del progetto

- Applicare tecniche di Data Cleaning
- Effettuare analisi esplorativa dei dati
- Confrontare algoritmi di classificazione
- Valutare le prestazioni dei modelli
- Interpretare i risultati tramite SHAP
- Comprendere quali fattori influenzano maggiormente il rischio cardiovascolare

---

## 📚 Competenze dimostrate

- Data Preprocessing
- Data Visualization
- Supervised Machine Learning
- Model Evaluation
- Explainable AI (XAI)
- Python Data Science Stack

---

## ⚠️ Disclaimer

Questo progetto è stato sviluppato esclusivamente a scopo didattico e di ricerca.

I risultati ottenuti non costituiscono una diagnosi medica e non devono essere utilizzati per prendere decisioni cliniche.

---

## 👨‍💻 Autore

**Emiliano Di Guglielmo**

GitHub: https://github.com/EmilianoDG5

Studente di Informatica con interesse per:

- Machine Learning
- Data Science
- Cybersecurity
- Software Development
