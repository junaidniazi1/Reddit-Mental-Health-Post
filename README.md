

# 🧠 Reddit Mental Health Post Classification (ML Project)

This project uses machine learning (Logistic Regression and Naive Bayes) to classify Reddit posts into mental health categories like **Depression**, **Anxiety**, **Bipolar**, **BPD**, **Schizophrenia**, and **Mental Illness**. The dataset was preprocessed, vectorized using TF-IDF, balanced using oversampling, and evaluated using classification metrics.

---

## 📂 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [ROC & AUC](#roc--auc)
- [Predictions on Custom Input](#predictions-on-custom-input)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Results](#results)
- [Model Saving & Loading](#model-saving--loading)
- [Contributors](#contributors)
- [License](#license)

---

## 📘 Project Overview

The goal of this project is to detect the **mental health topic** discussed in a Reddit post using Natural Language Processing and Machine Learning. This can support early detection and topic-specific routing of posts in mental health forums.

---

## 📊 Dataset

- **Source**: Reddit posts from 6 mental health-related subreddits.
- **Classes**:
  - `0`: BPD
  - `1`: bipolar
  - `2`: depression
  - `3`: Anxiety
  - `4`: schizophrenia
  - `5`: mentalillness

- **Sample Distribution (Before Balancing)**:
```

BPD:           229194
bipolar:       164216
depression:    149308
Anxiety:        45118
mentalillness:  42123
schizophrenia:  17619

````

---

## 🧹 Preprocessing

1. Removed nulls and duplicates
2. Cleaned text: lowercased, removed punctuation, stopwords, and URLs
3. Tokenized and lemmatized text (optional)
4. Used TF-IDF vectorization:
 ```python
 vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=5, max_df=0.7)
````

5. Applied `RandomOverSampler` to handle class imbalance.

---

## 🧠 Model Training

### 1. Logistic Regression

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

### 2. Multinomial Naive Bayes

```python
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
```

---

## 📈 Evaluation

* Used metrics: `Precision`, `Recall`, `F1-score`, and `Accuracy`
* Used macro and weighted averages
* Visualized via bar charts and confusion matrices

---

## 📉 ROC & AUC Curve

Plotted ROC-AUC curve for multi-class classification using `OneVsRestClassifier` and `label_binarize()` from sklearn.

---

## 🎯 Predictions on Custom Input

```python
text = input("Enter post text: ")
predicted_class = model.predict(vectorizer.transform([text]))
```

Mapped output class to:

```python
label_map = {
    0: 'BPD',
    1: 'bipolar',
    2: 'depression',
    3: 'Anxiety',
    4: 'schizophrenia',
    5: 'mentalillness'
}
```

---

## 💾 Model Saving & Loading

```python
import joblib

# Save
joblib.dump(model, "logistic_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Load
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
```

---

## ⚙️ Installation & Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/reddit-mental-health-classifier.git
   cd reddit-mental-health-classifier
   ```

2. Create environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

```bash
python main.py
```

Or, for notebook users:

```bash
jupyter notebook mental_health_classifier.ipynb
```

---

## 📊 Results

| Model              | Accuracy | F1-Score (Macro Avg) |
| ------------------ | -------- | -------------------- |
| LogisticRegression | \~71%    | \~0.61               |
| Naive Bayes        | \~65%    | \~0.55               |

---

## 👨‍💻 Contributors

* [Your Name](https://github.com/yourusername)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📌 Future Improvements

* Use LSTM / BERT for deeper learning
* Streamlit or Flask UI for live predictions
* Better text preprocessing and embeddings (word2vec or transformers)

```

---

### 📝 Next Steps

- Save it as `README.md` and place it in your project root.
- Add `requirements.txt` (I can generate that too).
- Let me know if you want to build a **Streamlit web app UI**.

Would you like me to prepare that `requirements.txt` and Streamlit interface too?
```
