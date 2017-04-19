from pyspark import SparkContext
import gc
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.mllib.evaluation import RankingMetrics

def recommend(m,x,num=10):
	return m.recommendProducts(x,num)

if __name__ == "__main__":
    sc = SparkContext()
	data = sc.textFile("train_data.txt",1)
	data=data.map(lambda x:x.split('\t'))
	valid_data = sc.textFile("valid_visible.txt",1)
	valid_data=valid_data.map(lambda x:x.split('\t'))
	valid_productid = valid_data.map(lambda x: x[1]).zipWithIndex().map(lambda x:x[1])
	valid_userid = valid_data.map(lambda x: x[0]).zipWithIndex().map(lambda x:x[1])
	data = data.union(valid_data)
	userid = data.map(lambda x: x[0]).zipWithIndex().map(lambda x:x[1])
	productid = data.map(lambda x: x[1]).zipWithIndex().map(lambda x:x[1])
	rating = data.map(lambda x:x[2])
	als_data = userid.zip(productid).zip(rating)
	ratings = als_data.map(lambda l: Rating(int(l[0][0]), int(l[0][1]), float(l[1])))
	del data
	del userid
	del productid
	del rating
	del als_data
	gc.collect()
	rank = 50
	numIterations = 10
	model = ALS.trainImplicit(ratings, rank, numIterations, alpha=0.01)
	model.save(sc, "target/tmp/myCollaborativeFilter")
	recommendation = valid_userid.map(lambda x:model.recommendProducts(x,10))
	valid_product = valid_userid.zip(valid_productid).groupByKey().map(lambda x:x[1])
	predictionAndLabels = valid_product.zip(recommendation)
	metrics = RankingMetrics(predictionAndLabels)
	precision = metrics.precisionAt(5)





	model.save(sc, "target/tmp/myCollaborativeFilter")
	sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
	testdata = ratings.map(lambda p: (p[0], p[1]))
	predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
	ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
	MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
	print("Mean Squared Error = " + str(MSE))











	#Examine the latent features for one product
model.productFeatures().first()
#(12, array('d', [-0.29417645931243896, 1.8341970443725586, 
    #-0.4908868968486786, 0.807500958442688, -0.8945541977882385]))

#Examine the latent features for one user
model.userFeatures().first()
#(12, array('d', [1.1348751783370972, 2.397622585296631,
    #-0.9957215785980225, 1.062819480895996, 0.4373367130756378]))

# For Product X, Find N Users to Sell To
model.recommendUsers(242,100)

# For User Y Find N Products to Promote
model.recommendProducts(196,10)

#Predict Single Product for Single User
model.predict(196, 242)

