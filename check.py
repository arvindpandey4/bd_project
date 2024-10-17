import requests

api_key = "328e081f35d447f68c89c4841a470ef6"
url = "https://newsapi.org/v2/everything"
parameters = {
    'q': 'India',   
    'language': 'en', 
    'sortBy': 'publishedAt', 
    'apiKey': api_key
}


response = requests.get(url, params=parameters)

if response.status_code == 200:
    print("API Request Successful!")
    news_data = response.json()
    

    if news_data['totalResults'] == 0:
        print("No articles found!")
    else:
        print(f"Total articles found: {news_data['totalResults']}")
        articles = news_data['articles']  # Extract the list of articles
        
        for article in articles[:5]:  # Print the first 5 articles
            print(f"Title: {article['title']}")
            print(f"Content: {article['content']}\n")
else:
    print(f"API Request Failed with status code: {response.status_code}")
