import requests
import re
import nltk
import pandas as pd
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

api_key = "328e081f35d447f68c89c4841a470ef6"
url = "https://newsapi.org/v2/everything"
parameters = {
    'q': 'India',
    'language': 'en',
    'sortBy': 'publishedAt',
    'apiKey': api_key
}

def fetch_news():
    try:
        response = requests.get(url, params=parameters)
        response.raise_for_status()
        news_data = response.json()

        if news_data['totalResults'] == 0:
            print("No articles found!")
            return []
        else:
            print(f"Total articles found: {news_data['totalResults']}")
            return news_data['articles']

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    clean_text = ' '.join(filtered_tokens)
    return clean_text

def process_articles(articles):
    cleaned_articles = []
    for article in articles:
        content = article.get('content')
        if content:
            try:
                language = detect(content)
                if language == 'en':
                    cleaned_content = preprocess_text(content)
                    cleaned_articles.append({
                        'title': article['title'],
                        'cleaned_content': cleaned_content
                    })
            except Exception as e:
                print(f"Language detection failed for article: {article.get('title', 'Unknown Title')}. Error: {e}")
    return cleaned_articles

def save_to_csv(cleaned_articles, file_name):
    df = pd.DataFrame(cleaned_articles)
    df.to_csv(file_name, index=False)
    print(f"Data saved to {file_name}")

def main():
    articles = fetch_news()
    cleaned_articles = process_articles(articles)
    if cleaned_articles:
        save_to_csv(cleaned_articles, 'cleaned_articles.csv')

if __name__ == "__main__":
    main()
