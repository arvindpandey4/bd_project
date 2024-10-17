import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Load dataset
df = pd.read_csv('News_Articles_Indian_Express.csv')

# Inspect column names
print("Columns in the dataset:", df.columns)

# Strip any possible whitespace in column names
df.columns = df.columns.str.strip()

# Define feature (headline) and target (article_type)
X = df['headline'].fillna('')  # Handling missing values in headlines, if any
y = df['article_type']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression Model
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_tfidf, y_train)
y_pred_lr = logistic_regression_model.predict(X_test_tfidf)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr, output_dict=True)

# Multinomial Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb, output_dict=True)

# Metrics comparison
labels = ['Logistic Regression', 'Multinomial NB']
accuracies = [accuracy_lr, accuracy_nb]
precisions = [report_lr['weighted avg']['precision'], report_nb['weighted avg']['precision']]
recalls = [report_lr['weighted avg']['recall'], report_nb['weighted avg']['recall']]
f1_scores = [report_lr['weighted avg']['f1-score'], report_nb['weighted avg']['f1-score']]

# Bar chart for performance metrics
plt.figure(figsize=(8, 5))
x = np.arange(len(labels))
width = 0.2

plt.bar(x - width, accuracies, width, label='Accuracy')
plt.bar(x, precisions, width, label='Precision')
plt.bar(x + width, recalls, width, label='Recall')
plt.bar(x + 2*width, f1_scores, width, label='F1 Score')

plt.xticks(x, labels)
plt.ylabel('Score')
plt.title('Model Performance Metrics')
plt.ylim(0, 1)
plt.legend()
plt.show()

# Confusion matrices
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_nb = confusion_matrix(y_test, y_pred_nb)

# Plot confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].imshow(cm_lr, cmap='Blues')
ax[0].set_title('Logistic Regression Confusion Matrix')
ax[0].set_xticks(np.arange(len(np.unique(y_test))))
ax[0].set_yticks(np.arange(len(np.unique(y_test))))

ax[1].imshow(cm_nb, cmap='Greens')
ax[1].set_title('Naive Bayes Confusion Matrix')
ax[1].set_xticks(np.arange(len(np.unique(y_test))))
ax[1].set_yticks(np.arange(len(np.unique(y_test))))

plt.show()
