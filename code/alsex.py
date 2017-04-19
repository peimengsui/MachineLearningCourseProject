import os
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.mllib.evaluation import RankingMetrics

sqlContext = HiveContext(sc)




sc = SparkContext()
sqlContext = SQLContext(sc)

plays_df_schema = StructType(
  [StructField('Plays', IntegerType()),
   StructField('songId', StringType()),
   StructField('userId', StringType()),
   ]
)


als_data = sc.textFile("train_triplets.txt",1)
als_data=als_data.map(lambda x:x.split('\t'))
als_data = als_data.map(lambda x: Row(userID=x[0], songID=x[1],Plays=int(x[2])))
raw_plays_df = sqlContext.createDataFrame(als_data,schema=plays_df_schema)
userId_change = raw_plays_df.select('userId').distinct().select('userId', F.monotonically_increasing_id().alias('new_userId'))
songId_change = raw_plays_df.select('songId').distinct().select('songId', F.monotonically_increasing_id().alias('new_songId'))

unique_users = userId_change.count()
unique_songs = songId_change.count()
print('Number of unique users: {0}'.format(unique_users))
print('Number of unique songs: {0}'.format(unique_songs))


raw_plays_df_with_int_ids = raw_plays_df.join(userId_change, 'userId').join(songId_change, 'songId')


raw_plays_df_with_int_ids.cache()
raw_plays_df_with_int_ids.show(5)


# We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing
seed=1
(split_60_df, split_a_20_df, split_b_20_df) = raw_plays_df_with_int_ids.randomSplit([0.6, 0.2, 0.2], seed = seed)

# Let's cache these datasets for performance
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()

print('Training: {0}, validation: {1}, test: {2}\n'.format(
  training_df.count(), validation_df.count(), test_df.count())
)
training_df.show(3)
validation_df.show(3)
test_df.show(3)

validation_df = validation_df.withColumn("Plays", validation_df["Plays"].cast(DoubleType()))
validation_df = validation_df.withColumn("new_userId", validation_df["new_userId"].cast('integer'))
validation_df = validation_df.withColumn("new_songId", validation_df["new_songId"].cast('integer'))
validation_df.show(10)

als_data = validation_df.select('new_userId','new_songId','Plays').rdd.map(tuple)

ratings = als_data.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[1])))

rank = 50
numIterations = 10
model = ALS.trainImplicit(ratings, rank, numIterations, alpha=0.15)
#model.save(sc, "target/tmp0418/myCollaborativeFilter")
valid_userid=validation_df.select('new_userId').rdd.map(tuple)
recommendation = model.recommendProductsForUsers(500).map(lambda x:x[1]).map(lambda x:[r.product for r in x])
valid_list=als_data.map(lambda x:(x[0],[x[1]])).reduceByKey(lambda a, b: a + b).map(lambda x:list(set(x[1])))



valid_list = validation_df.select('new_userId','new_songId').groupby("new_userId").agg(F.collect_list("new_songId"))

als_data = validation_df.select('new_userId','new_songId','Plays')
ratings = als_data.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[1])))

rank = 50
numIterations = 10
model = ALS.trainImplicit(ratings, rank, numIterations, alpha=0.15)
model.save(sc, "target/tmp/myCollaborativeFilter")
recommendation = valid_userid.map(lambda x:recommend(model,x,10))
valid_product = valid_userid.zip(valid_productid).groupByKey().map(lambda x:x[1])
predictionAndLabels = valid_product.zip(recommendation)
metrics = RankingMetrics(predictionAndLabels)
precision = metrics.precisionAt(5)



# Let's initialize our ALS learner
als = ALS(implicitPrefs=True)

# Now set the parameters for the method
als.setMaxIter(5)\
   .setSeed(seed)\
   .setItemCol("new_songId")\
   .setRatingCol("Plays")\
   .setUserCol("new_userId")

# Now let's compute an evaluation metric for our test dataset
# We Create an RMSE evaluator using the label and predicted columns
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="Plays", metricName="rmse")

tolerance = 0.03
ranks = [4, 8, 12, 16]
regParams = [0.15, 0.2, 0.25]
errors = [[0]*len(ranks)]*len(regParams)
models = [[0]*len(ranks)]*len(regParams)
err = 0
min_error = float('inf')
best_rank = -1
i = 0

for regParam in regParams:
	j = 0
	for rank in ranks:
  # Set the rank here:
		als.setParams(rank = rank, regParam = regParam,implicitPrefs=True)
		# Create the model with these parameters.
		model = als.fit(training_df)
		# Run the model to create a prediction. Predict against the validation_df.
		predict_df = model.transform(validation_df)

		# Remove NaN values from prediction (due to SPARK-14489)
		predicted_plays_df = predict_df.filter(predict_df.prediction != float('nan'))
		predicted_plays_df = predicted_plays_df.withColumn("prediction", F.abs(F.round(predicted_plays_df["prediction"],0)))
		# Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
		error = reg_eval.evaluate(predicted_plays_df)
		errors[i][j] = error
		models[i][j] = model
		print ('For rank %s, regularization parameter %s the RMSE is %s' % (rank, regParam, error))
		if error < min_error:
			min_error = error
			best_params = [i,j]
		j += 1
	i += 1

als.setRegParam(0.15)
als.setRank(12)
print ('The best model was trained with regularization parameter %s' % regParams[best_params[0]])
print ('The best model was trained with rank %s' % ranks[best_params[1]])
my_model = models[best_params[0]][best_params[1]]


test_df = test_df.withColumn("Plays", test_df["Plays"].cast(DoubleType()))
predict_df = my_model.transform(test_df)

# Remove NaN values from prediction (due to SPARK-14489)
predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))

# Round floats to whole numbers
predicted_test_df = predicted_test_df.withColumn("prediction", F.abs(F.round(predicted_test_df["prediction"],0)))
# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_df DataFrame
test_RMSE = reg_eval.evaluate(predicted_test_df)

print('The model had a RMSE on the test set of {0}'.format(test_RMSE))

