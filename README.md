# Twitter Sentiment Analysis using Machine Learning & NLP

A complete Natural Language Processing (NLP) and Machine Learning project that classifies tweets as **Positive** or **Negative**.

The project uses a large Twitter sentiment dataset containing approximately **1.6 million tweets**, applies NLP-based text preprocessing, converts text into numerical features using **TF-IDF**, and trains a **Logistic Regression** classifier for sentiment prediction.

---

## Project Overview

Social media contains a huge amount of opinion-rich text. Manually analyzing millions of tweets is impossible, so this project builds an automated sentiment classification system.

Given a tweet such as:

> `I absolutely love this product!`

the model predicts:

```text
Positive Tweet
```

For a tweet such as:

> `This is the worst experience ever.`

the model predicts:

```text
Negative Tweet
```

---

## Problem Statement

Build a machine learning model capable of automatically identifying the sentiment expressed in a tweet.

### Input
A raw tweet/text message.

### Output

```text
0 → Negative
1 → Positive
```

The original dataset used sentiment labels `0` and `4`. During preprocessing, the labels were converted to a binary format:

```text
0 → 0  # Negative
4 → 1  # Positive
```

---

## Dataset

The project uses a Twitter sentiment dataset containing approximately:

**1,600,000 tweets**

The dataset contains sentiment labels and tweet text along with other metadata fields.

The large CSV dataset is intentionally **not included in this GitHub repository** because the file is approximately **227 MB**, exceeding GitHub's standard 100 MB file limit.

The dataset is kept locally and excluded through `.gitignore`.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Regular Expressions | Text cleaning |
| NLTK | NLP preprocessing |
| Scikit-learn | Feature extraction, splitting and ML |
| TF-IDF | Text feature extraction |
| Logistic Regression | Sentiment classification |
| Pickle | Saving/loading the trained model |
| Jupyter Notebook | Development and experimentation |
| Git & GitHub | Version control |

---

# Project Workflow

```text
Raw Twitter Dataset
        │
        ▼
Data Loading
        │
        ▼
Dataset Inspection
        │
        ├── Shape
        ├── Dataset Information
        ├── Missing Values
        ├── Duplicate Values
        └── Target Distribution
        │
        ▼
Text Preprocessing
        │
        ├── Convert to lowercase
        ├── Remove URLs
        ├── Remove @mentions
        ├── Remove non-alphabetic characters
        └── Remove extra spaces
        │
        ▼
TF-IDF Feature Extraction
        │
        ▼
Train / Test Split
        │
        ▼
Logistic Regression
        │
        ▼
Model Evaluation
        │
        ▼
Save Trained Model
        │
        ▼
Load Model
        │
        ▼
Predict Sentiment of New Tweets
```

---

# Exploratory Data Analysis

Before training the model, the dataset was explored to understand its structure and quality.

The following checks were performed:

### 1. Dataset Shape

The dataset contains approximately **1.6 million records**.

### 2. Dataset Information

Data types, non-null values and memory usage were inspected using:

```python
df.info()
```

### 3. Missing Values

Missing values were checked to ensure that the model would not receive incomplete text or labels.

### 4. Duplicate Values

Duplicate records were checked to understand whether repeated tweets were present.

### 5. Sentiment Distribution

The target distribution was visualized to understand the balance between positive and negative tweets.

---

# NLP Text Preprocessing

Twitter text contains URLs, mentions, special characters and unnecessary spaces that can introduce noise into a machine learning model.

A custom cleaning function was created using regular expressions.

```python
import re

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
```

The cleaned text was stored in a new column:

```python
twitter_data["clean_text"] = twitter_data["text"].apply(clean_text)
```

---

# Stemming

Stemming was explored as part of the NLP preprocessing stage using NLTK's `PorterStemmer`.

However, applying stemming word-by-word across approximately 1.6 million tweets is computationally expensive.

Therefore, the project prioritizes efficient preprocessing and TF-IDF feature extraction rather than unnecessarily performing expensive stemming across the entire dataset.

This is an important practical trade-off when working with large-scale text data.

---

# Feature Extraction — TF-IDF

Machine learning algorithms cannot directly understand raw text.

TF-IDF (**Term Frequency–Inverse Document Frequency**) converts text into numerical feature vectors based on the importance of words within the dataset.

Conceptually:

```text
Tweet
   ↓
Clean Text
   ↓
TF-IDF Vectorizer
   ↓
Numerical Feature Vector
```

The vectorizer is fitted on the training data:

```python
X_train_vec = Vectorizer.fit_transform(X_train)
```

and the same fitted vectorizer is used to transform test data:

```python
X_test_vec = Vectorizer.transform(X_test)
```

This prevents information from the test set from leaking into the training process.

---

# Train-Test Split

The dataset was divided into training and testing portions using `train_test_split`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2,
    stratify=y
)
```

### Split

- **80%** → Training
- **20%** → Testing
- `random_state = 2`
- Stratification was used to preserve the sentiment distribution.

---

# Machine Learning Model

## Logistic Regression

Logistic Regression was selected as the classification algorithm.

```python
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(
    X_train_vec,
    y_train
)
```

Logistic Regression is a strong baseline for high-dimensional sparse text classification because TF-IDF produces a large sparse feature matrix.

---

# Model Evaluation

The trained model was evaluated using accuracy.

The observed training accuracy during development was approximately:

```text
81.27%
```

> **Note:** This value is the training accuracy observed in the notebook. It should not be presented as the final test-set/generalization accuracy. The test-set evaluation should be reported separately if calculated.

---

# Sentiment Prediction

After training, the model can classify individual tweets.

Example workflow:

```python
X_new = X_test[i]

new_tweet_vec = Vectorizer.transform([X_new])

prediction = loaded_model.predict(new_tweet_vec)

if prediction[0] == 0:
    print("Negative Tweet")
else:
    print("Positive Tweet")
```

The index can be changed dynamically:

```python
i = int(input("Enter tweet index: "))
```

This allows individual tweets from the test set to be inspected and classified.

---

# Model Persistence

The trained Logistic Regression model was saved using Python's `pickle` module.

```python
import pickle

with open("Trained_model.pkl", "wb") as file:
    pickle.dump(lr_model, file)
```

The saved model can later be loaded without retraining:

```python
with open("Trained_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)
```

Because the Logistic Regression model expects TF-IDF features, the fitted TF-IDF vectorizer must also be available when making predictions from raw text.

---

# Project Structure

```text
Sentiment-Analysis/
│
├── sentiment_analysis.ipynb
├── Trained_model.pkl
├── README.md
├── .gitignore
│
└── sentiment.csv          # Not included in GitHub
```

---

# How to Run the Project

## 1. Clone the repository

```bash
git clone https://github.com/sahilrawat1415-cell/Sentiment-Analysis.git
```

## 2. Move into the project directory

```bash
cd Sentiment-Analysis
```

## 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn jupyter
```

## 4. Download required NLTK resources

If using the NLTK components from the notebook:

```python
import nltk

nltk.download("stopwords")
nltk.download("punkt")
```

## 5. Add the dataset

Place the Twitter sentiment CSV dataset in the project directory locally.

The dataset is excluded from GitHub because of its size.

## 6. Open the notebook

```bash
jupyter notebook sentiment_analysis.ipynb
```

Run the notebook cells in order.

---

# Key Learning Outcomes

This project demonstrates practical experience with:

- Working with a **large-scale text dataset**
- Exploratory Data Analysis
- Data quality checking
- Handling categorical sentiment labels
- Regular-expression based text cleaning
- NLP preprocessing
- Stopword handling
- Stemming concepts and computational trade-offs
- TF-IDF feature extraction
- Train-test splitting
- Logistic Regression for text classification
- Model evaluation
- Model serialization using Pickle
- Loading a trained model
- Making predictions on individual tweets
- Git/GitHub project management

---

# Challenges Faced

### Large Dataset

Processing approximately 1.6 million tweets requires considerably more memory and computation than a small NLP dataset.

### NLP Processing Time

Word-by-word stemming across the entire dataset was found to be time-consuming. This led to a more efficient preprocessing approach.

### Large GitHub File

The original CSV dataset was approximately 227 MB, which exceeded GitHub's 100 MB file limit.

The dataset was therefore excluded from version control.

### Text-to-Feature Conversion

A trained Logistic Regression model cannot directly consume raw text. The same fitted TF-IDF transformation used during training is required before prediction.

---

# Future Improvements

The project can be extended with:

- Precision, Recall and F1-score
- Confusion Matrix
- ROC-AUC evaluation
- Hyperparameter tuning
- Comparison with Naive Bayes
- Comparison with Linear SVM
- Better Twitter-specific preprocessing
- N-gram optimization
- Model pipeline combining TF-IDF + classifier
- REST API using Flask/FastAPI
- Web interface using HTML/CSS/React
- Cloud deployment
- Real-time tweet sentiment prediction

---

# Conclusion

This project demonstrates an end-to-end **NLP + Machine Learning workflow** for sentiment classification.

Starting from approximately **1.6 million raw tweets**, the project performs data exploration and cleaning, transforms textual data into numerical TF-IDF features, trains a Logistic Regression classifier, evaluates the model, saves the trained model, and performs sentiment predictions.

The project also highlights an important real-world machine learning lesson: **efficient preprocessing and proper separation of training-time transformations from prediction-time transformations are just as important as choosing the model itself.**

---

## Author

**Sahil Rawat**

GitHub:  
https://github.com/sahilrawat1415-cell

---

## License

This project is intended for educational and portfolio purposes.
