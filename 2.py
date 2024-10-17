import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, TFBertModel, logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Suppress warnings from Hugging Face
logging.set_verbosity_error()

# Load dataset
df = pd.read_csv('News_Articles_Indian_Express.csv')

# Use 'headline' for the text and 'article_type' for the category/label
X = df['headline'].fillna('')  # Handling any missing values in headlines
y = df['article_type']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IndicBERT Tokenization and Embeddings
try:
    tokenizer = BertTokenizer.from_pretrained('ai4bharat/indic-bert')
    model = TFBertModel.from_pretrained('ai4bharat/indic-bert')
except Exception as e:
    print("Error loading the IndicBERT model:", e)
    # You might want to exit or use a fallback model here
    exit()

# Tokenizing the training and test data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128, return_tensors='tf')
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128, return_tensors='tf')

# Use CLS token's embedding as feature representation
X_train_bert = model(train_encodings).last_hidden_state[:, 0, :]
X_test_bert = model(test_encodings).last_hidden_state[:, 0, :]

# Logistic Regression on IndicBERT Embeddings
lr_bert_model = LogisticRegression(max_iter=1000)
lr_bert_model.fit(X_train_bert.numpy(), y_train)
y_pred_bert = lr_bert_model.predict(X_test_bert.numpy())
accuracy_bert = accuracy_score(y_test, y_pred_bert)
report_bert = classification_report(y_test, y_pred_bert, output_dict=True)

# Performance metrics
labels = ['IndicBERT + LR']
accuracies = [accuracy_bert]
precisions = [report_bert['weighted avg']['precision']]
recalls = [report_bert['weighted avg']['recall']]
f1_scores = [report_bert['weighted avg']['f1-score']]

# Bar chart for performance metrics
plt.figure(figsize=(6, 5))
x = np.arange(len(labels))
width = 0.2

plt.bar(x - width, accuracies, width, label='Accuracy')
plt.bar(x, precisions, width, label='Precision')
plt.bar(x + width, recalls, width, label='Recall')
plt.bar(x + 2*width, f1_scores, width, label='F1 Score')

plt.xticks(x, labels)
plt.ylabel('Score')
plt.title('IndicBERT + LR Model Performance Metrics')
plt.ylim(0, 1)
plt.legend()
plt.show()

# Confusion matrix for IndicBERT
cm_bert = confusion_matrix(y_test, y_pred_bert)

# Plot confusion matrix
plt.imshow(cm_bert, cmap='Oranges')
plt.title('IndicBERT + LR CM')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(ticks=np.arange(len(np.unique(y))), labels=np.unique(y))
plt.yticks(ticks=np.arange(len(np.unique(y))), labels=np.unique(y))
plt.show()
