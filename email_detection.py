import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Sample spam/ham emails (increase data for more realistic results)
data = {
    'text': ['Free money now!!!', 'Hi, how are you?', 'Limited time offer!!! Act now!', 
             'Lunch at 12?', 'You have won a prize!', 'Meeting tomorrow', 'Free trial for you', 
             'Claim your free gift card!', 'Let’s catch up later', 'Earn cash fast, click here!', 
             'Are you coming to the party?', 'Win big with this simple trick', 'Let’s have dinner tomorrow', 
             'Congratulations! You’ve been selected for a prize', 'Free subscription for the first month',
             'Can we reschedule our meeting?', 'Get rich quick!', 'Join us for a webinar tomorrow', 
             'This is your last chance to claim your reward', 'See you at the conference',
             'Click here for free rewards!', 'Special offer: Buy now and save 50%', 
             'Hello, are you available for a meeting?', 'Your account has been compromised', 
             'Act fast! Limited stock available', 'Family dinner this weekend?', 
             'Win $1000 instantly! Click here', 'Secure your account now', 
             'Exclusive deal: Free shipping for members', 'Can you send me the report?'],
    'label': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 
              1, 1, 0, 1, 1, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

# Preprocessing function to clean text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)

# Convert text data into TF-IDF feature vectors
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression with hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
model = LogisticRegression()
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

# Make predictions
y_pred = grid.best_estimator_.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve and AUC Score
y_pred_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot ROC Curve
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
