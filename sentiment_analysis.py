from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder \
    .appName("NewsSentimentAnalysis") \
    .getOrCreate()


articles_df = spark.read.csv('cleaned_articles.csv', header=True, inferSchema=True)


tokenizer = Tokenizer(inputCol="cleaned_content", outputCol="words")
tokenized_df = tokenizer.transform(articles_df)


remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered_df = remover.transform(tokenized_df)


hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
featurized_data = hashing_tf.transform(filtered_df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(featurized_data)
rescaled_data = idf_model.transform(featurized_data)


labeled_data = rescaled_data.withColumn("sentiment", rescaled_data['title'].rlike("positive|good").cast("integer"))


train_data, test_data = labeled_data.randomSplit([0.8, 0.2], seed=1234)


lr = LogisticRegression(labelCol="sentiment", featuresCol="features", maxIter=10)
lr_model = lr.fit(train_data)


predictions = lr_model.transform(test_data)
predictions.select("title", "prediction", "sentiment").show(10)


evaluator = MulticlassClassificationEvaluator(labelCol="sentiment", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test set accuracy = {accuracy}")


spark.stop()
