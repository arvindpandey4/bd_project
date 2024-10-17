from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Data Ingestion") \
    .getOrCreate()

df = spark.read.csv("C:\Arvind\Projects\BD\india-news-headlines.csv", header=True)
df.show(5)

df = df.filter(col('article').isNotNull())
df.write.format("csv").save("/path_to_save/cleaned_news_articles.csv")
