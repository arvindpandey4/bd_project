import requests

api_key = "328e081f35d447f68c89c4841a470ef6"
url = "https://newsapi.org/v2/top-headlines"
parameters = {
    'country': 'in',    
    'language': 'en',  
    'category': 'general',   
    'apiKey': api_key
}


response = requests.get(url, params=parameters)


news_data = response.json()
articles = news_data['articles']


for article in articles:
    print(f"Title: {article['title']}")
    print(f"Description: {article['description']}")
    print(f"Content: {article['content']}\n")
