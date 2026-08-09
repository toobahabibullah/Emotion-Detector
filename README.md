# Emotion Detector

A machine learning project that classifies text into one of three emotions: **Anger, Joy, or Fear**.

The project uses **Natural Language Processing (NLP)** techniques to convert text into numerical features and a **Multinomial Naive Bayes** classifier to predict the emotion expressed in a sentence.

## Project Overview

The goal of this project is to build a simple text-based emotion detection system.

Given a text such as:

> "I am so happy today!"

the model predicts:

```text
Predicted Emotion: joy
```

The model can classify text into three emotion categories:

* 😡 **Anger**
* 😊 **Joy**
* 😨 **Fear**

## Dataset

The project uses a dataset containing **5,937 text samples** with two columns:

| Column    | Description                            |
| --------- | -------------------------------------- |
| `Comment` | Text containing an emotional statement |
| `Emotion` | Emotion label associated with the text |

The dataset contains three emotion classes:

* `anger` — 2,000 samples
* `joy` — 2,000 samples
* `fear` — 1,937 samples

The dataset contains no null values after preprocessing and duplicate records are removed before training.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Regular Expressions (Regex)

## Machine Learning Pipeline

The project follows these steps:

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Remove Null Values
   ↓
Remove Duplicates
   ↓
Text Preprocessing
   ↓
CountVectorizer
   ↓
Train/Test Split
   ↓
Multinomial Naive Bayes
   ↓
Model Evaluation
   ↓
Emotion Prediction
```

## Data Preprocessing

The text data is cleaned before training.

The preprocessing includes:

1. Converting text to lowercase.
2. Removing non-alphabetic characters.
3. Removing null values.
4. Removing duplicate records.

Example:

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text
```

## Feature Extraction

`CountVectorizer` is used to convert the cleaned text into numerical features that can be understood by the machine learning model.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data['Comment'])
y = data['Emotion']
```

## Machine Learning Model

The project uses **Multinomial Naive Bayes**, a commonly used algorithm for text classification.

The data is divided into:

* **80% training data**
* **20% testing data**

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = MultinomialNB()
model.fit(X_train, y_train)
```

## Model Performance

The model achieved approximately **89% accuracy** on the test set. The classification report shows an F1-score of approximately **0.89 for all three emotions**.

### Classification Results

| Emotion              | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| Anger                |      0.88 |   0.90 |     0.89 |
| Fear                 |      0.88 |   0.90 |     0.89 |
| Joy                  |      0.92 |   0.87 |     0.89 |
| **Overall Accuracy** |           |        | **0.89** |

The model's test accuracy was `0.8914` in the notebook.

## Confusion Matrix

A confusion matrix is generated to analyze how well the model distinguishes between:

* Anger
* Joy
* Fear

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["anger", "joy", "fear"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["anger", "joy", "fear"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()
```

One observation from the evaluation is that **anger and fear can sometimes be confused**, since both represent strong negative emotions.

## Example Predictions

The trained model was tested on new, unseen text:

```text
Text: I am so happy today!
Predicted Emotion: joy

Text: I am furious at what happened
Predicted Emotion: anger

Text: Something feels very wrong and scary
Predicted Emotion: fear
```

These examples demonstrate how the trained model can be used to classify new text.

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd emotion-detector
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### 4. Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and run the cells sequentially.

## Project Structure

```text
emotion-detector/
│
├── Emotion_classify_Data[1].csv
├── Emotion_Detector.ipynb
├── README.md
└── requirements.txt
```

> File names may be changed according to the actual names used in your repository.

## Future Improvements

The current project provides a strong baseline for text emotion classification. It could be further improved by:

* Using **TF-IDF** instead of only word counts.
* Testing algorithms such as Logistic Regression, SVM, or Random Forest.
* Using **word embeddings**.
* Trying transformer-based NLP models.
* Adding more emotion categories.
* Building a web interface for real-time predictions.
* Saving the trained model with `joblib` or `pickle`.
* Creating an API using FastAPI.
* Adding confidence scores to predictions.

## Key Takeaways

* The dataset contains **5,937 labeled text samples**.
* Three emotions are classified: **anger, joy, and fear**.
* Text is converted into numerical features using **CountVectorizer**.
* **Multinomial Naive Bayes** is used for classification.
* The model achieves approximately **89% accuracy**.
* The model achieves approximately **0.89 F1-score** across all three emotion classes.

## Author

**Tooba Habibullah**

BS Artificial Intelligence Student

