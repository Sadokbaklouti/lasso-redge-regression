# 📨 YouTube Comment Spam Detection: Naive Bayes (NLP)

> **"An original project concept inspired by 365 Data Science."**

This project builds a **Natural Language Processing (NLP)** text classification model to automatically filter out spam comments from YouTube video comment sections. It implements a **Multinomial Naive Bayes** classifier trained on a collection of tokenized text sequences to classify comments as either legitimate (**Ham**) or advertisements/malicious links (**Spam**).

---

## 📊 Dataset & Architecture

The dataset consists of 5 separate CSV files containing a total of **1,955 comments** harvested from popular YouTube videos (such as music videos from Eminem, Psy, and Katy Perry). 

### Data Preprocessing & NLP Pipeline:
1. **Dynamic Loading**: Extracted and concatenated all 5 CSV data frames seamlessly from a compressed `.zip` file archive.
2. **Feature Pruning**: Retained only the text message (`CONTENT`) and target classification (`CLASS`). Dropped unnecessary metadata (`COMMENT_ID`, `AUTHOR`, `DATE`).
3. **Bag-of-Words Vectorization**: Utilized Scikit-Learn's `CountVectorizer` to tokenize phrases, build a localized vocabulary of **4,453 unique tokens**, and transform textual strings into numerical sparse frequency arrays.
4. **Stratified Splitting**: Divided data into training (80%) and testing (20%) partitions while enforcing stratification to keep balanced target proportions across both sets.

---

## 🛠️ Machine Learning Model

Because text classification involves discrete word frequency counts, a **Multinomial Naive Bayes (MultinomialNB)** algorithm was leveraged. Naive Bayes is exceptionally lightweight, computationally fast, and highly effective for baseline text filtering problems.

---

## 📈 Model Evaluation Metrics

The classifier was evaluated on an unseen test dataset consisting of 391 comments. The performance metrics demonstrate outstanding filtering capability:

### 1. Classification Report
The model achieves a stellar overall **Accuracy of 94%**. Both classes exhibit incredibly well-balanced performance attributes:

* **Ham (Class 0)**: High precision (**96%**), guaranteeing that legitimate user messages are rarely misclassified as spam.
* **Spam (Class 1)**: Outstanding recall (**96%**), demonstrating that the algorithm catches the vast majority of junk messages and unwanted promotions.

```text
              precision    recall  f1-score   support

         Ham       0.96      0.93      0.94       190
        Spam       0.93      0.96      0.95       201

    accuracy                           0.94       391
   macro avg       0.94      0.94      0.94       391
weighted avg       0.94      0.94      0.94       391
