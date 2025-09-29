# 📝 Toxic Comment Classification Challenge

This project is based on the **[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)** on Kaggle.  
The goal is to build a **multi-label text classification model** that detects different types of toxicity in online comments.

---

## 🚨 Problem Statement
Given a comment, the model classifies it into one or more categories of harmful language.

---

## 🏷 Toxicity Categories
The model predicts the following **six types of toxicity**:

- **toxic**: General disrespectful behavior  
- **severe_toxic**: Highly offensive language  
- **obscene**: Obscene or vulgar language  
- **threat**: Expressing intent to harm  
- **insult**: Direct personal attacks  
- **identity_hate**: Targeted attacks based on identity  

---

## 📂 Dataset
The dataset contains text comments extracted from **Wikipedia’s talk page edits**, with binary labels for each toxicity category.  

- **train.csv** → Labeled data for training  
- **test.csv** → Unlabeled data for predictions  

---

## ⚙️ Workflow
The analysis follows a classic **NLP pipeline**:

1. **Data Exploration and Analysis**  
2. **Text Preprocessing**:  
   - Cleaning text  
   - Stemming (Porter Stemmer)  
   - Removing English stopwords  
3. **Feature Extraction**:  
   - TF-IDF Vectorization  
   - (1, 2) N-gram range  
   - Maximum 10,000 features  
4. **Model Training** (multi-label using `OneVsRestClassifier`):  
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
5. **Model Evaluation**:  
   - Macro F1-score  
   - 3-fold Cross-Validation  

---

## 📦 Libraries Used
- **pandas**, **numpy**  
- **scikit-learn**  
- **xgboost**  
- **nltk**  

---

## 📊 Results
- Multiple models were trained and evaluated using **Macro F1-score**.  
- Preprocessing + TF-IDF features provided a **strong baseline**.  
- **Random Forest** achieved the most **stable and high-performing results** compared to more complex models.  



## 📌 Notes
- Download **NLTK resources** (like stopwords) before running preprocessing.  
- Code runs in **Kaggle notebooks**.  

---

## ⏭️ Future Improvements
- 🔧 **Hyperparameter Tuning**: Optimize Logistic Regression with GridSearchCV  
- 🧠 **Advanced Features**: Use pre-trained embeddings (e.g., BERT, GloVe)  
- 🤖 **Deep Learning**: Explore RNNs, CNNs, or Transformers for better sequence modeling  

---


