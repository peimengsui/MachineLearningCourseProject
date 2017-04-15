from pyspark import SparkContext
import gc
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

if __name__ == "__main__":
    sc = SparkContext()
	data = sc.textFile("train_data.txt",1)
	data=data.map(lambda x:x.split('\t'))
	valid_data = sc.textFile("valid_visible.txt",1)
	valid_data=valid_data.map(lambda x:x.split('\t'))
	data = data.union(valid_data)
	userid = data.map(lambda x: x[0]).zipWithIndex().map(lambda x:x[1])
	productid = data.map(lambda x: x[1]).zipWithIndex().map(lambda x:x[1])
	rating = data.map(lambda x:x[2])
	als_data = userid.zip(productid).zip(rating)
	ratings = als_data.map(lambda l: Rating(int(l[0][0]), int(l[0][1]), float(l[1])))
	del data
	del valid_data
	del userid
	del productid
	del rating
	del als_data
	gc.collect()
	rank = 10
	numIterations = 10
	model = ALS.trainImplicit(ratings, rank, numIterations, alpha=0.01)
	model.save(sc, "target/tmp/myCollaborativeFilter")
	sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
	testdata = ratings.map(lambda p: (p[0], p[1]))
	predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
	ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
	MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
	print("Mean Squared Error = " + str(MSE))

