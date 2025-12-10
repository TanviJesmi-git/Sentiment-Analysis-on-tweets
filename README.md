# **Sentiment Analysis on tweets (using BiLSTM)**

## **📌 Introduction**

This project focuses on building a **Sentiment Analysis Model** that classifies tweets as **positive** or **negative** using Deep Learning and Natural Language Processing (NLP) techniques.
The model is trained on the **Sentiment140** dataset, which contains **1.6 million tweets**, making it ideal for large-scale text sentiment prediction tasks.

The project demonstrates essential components of NLP pipelines, including data cleaning, tokenization, sequence modeling using LSTMs, and evaluation using performance metrics and visualizations.

---

## **✨ Features**

* ✔️ **Automated text preprocessing** (removal of URLs, mentions, hashtags, special characters)
* ✔️ **Stemming and stopword removal** using NLTK
* ✔️ **Deep Learning model** using **Bidirectional LSTM**
* ✔️ **Embedding layer** for dense word representation
* ✔️ **Train/Validation/Test split** for robust evaluation
* ✔️ **Performance visualization** (accuracy, loss plots)
* ✔️ **Confusion matrix & classification report**
* ✔️ **Custom prediction function** for classifying new tweets
* ✔️ **High accuracy** with regularization, dropout, and optimization techniques

---

## **📂 Dataset**

### **Sentiment140 Dataset (Kaggle)**

* **Total tweets:** 1,600,000
* **Labels:**

  * `0` → Negative
  * `4` → Positive (converted to `1` during preprocessing)

### **Dataset Contents**

* Original tweet text
* Polarity label
* Metadata (IDs, timestamps, query info — not used)

### **Preprocessing Steps**

* Remove URLs
* Remove mentions (`@username`)
* Remove hashtags (`#happy → happy`)
* Remove non-alphabetic characters
* Convert to lowercase
* Tokenization
* Stopword removal
* Stemming using **PorterStemmer**

The cleaned text is stored in a new column: **stemmed_content**.

---

### **Model Architecture**

| Layer                      | Description                                         |
| -------------------------- | --------------------------------------------------- |
| **Embedding (128-dim)**    | Converts tokens to dense vector representations     |
| **SpatialDropout1D (0.3)** | Regularization for embedding layer                  |
| **Bi-LSTM (128 units)**    | Captures bidirectional sequence context             |
| **Bi-LSTM (64 units)**     | Additional layer for deeper learning                |
| **Dense (32, ReLU)**       | Fully connected hidden layer with L2 regularization |
| **Dropout (0.5)**          | Prevents overfitting                                |
| **Dense (1, Sigmoid)**     | Output layer for binary classification              |

### **Hyperparameters**

* **Tokenizer vocabulary size:** 50,000
* **Sequence length:** 50 tokens
* **Optimizer:** Adam (lr = 0.0005)
* **Loss:** Binary Crossentropy
* **Batch size:** 512
* **Epochs:** 12
* **Callbacks:**

  * EarlyStopping
  * ReduceLROnPlateau

---

## **📊 Results**

### **Evaluation Metrics**

* **Accuracy:** *(Add final accuracy observed during training)*
* **Classification Report:**

  * Precision
  * Recall
  * F1-Score

### **Confusion Matrix**

Displays True Positive, True Negative, False Positive, False Negative counts.

### **Training Curves**

Two graphs:

* **Accuracy vs Epochs**
* **Loss vs Epochs**

These help visualize learning behavior and overfitting.

### **Sample Predictions**

```
Tweet: I absolutely love this new phone!
→ Positive 😀 (Confidence: 0.982)

Tweet: This is the worst movie ever.
→ Negative 😡 (Confidence: 0.876)
```

---

## **📚 References**

* Sentiment140 Dataset — Kaggle
* TensorFlow Documentation
* NLTK Stopword Corpus
* scikit-learn
* Research papers on **LSTM** and **Word Embeddings**
