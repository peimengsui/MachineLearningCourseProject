import os
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkContext
import pandas as pd
import numpy as np 
sc = SparkContext()
train = sc.textFile("als_visible.txt")
test = sc.textFile("als_predict.txt")

train_ratings = train.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
test_ratings = test.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
rank = 30
numIterations = 10
model = ALS.trainImplicit(train_ratings, rank, numIterations)
model.save(sc, "target/tmprank30/myCollaborativeFilter")
#sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")

test_users = test_ratings.map(lambda x: x.user).collect()
test_users = list(set(test_users))
test_users = test_users[0:10000]

recs={}
i=0
for u in test_users:
  i+=1
  rec = model.recommendProducts(u,200)
  recs[u]=list(map(lambda r: r[1],rec))
  if i%100==0:
    	print(i)

groundTruth = {}
test = np.loadtxt('als_predict.txt',delimiter=',')
#test = test[0:10000,:]
groundTruth = {}
cur_key = test[0,0].item()
groundTruth[cur_key] = []
for i in range(test.shape[0]):
  new_key = test[i,0].item()
  if new_key!=cur_key:
    groundTruth[new_key] = [test[i,1].item()]
    cur_key = new_key
  else:
    groundTruth[cur_key].append(test[i,1].item())
# test = pd.read_csv('als_predict.txt',header=None,index_col = 0)
# test.ix[:,1] = test.ix[:,1].astype(int)
# test = test.ix[:,0:1]
# grouped = test.groupby(test.index)
# df = grouped.aggregate(lambda x: list(x))

# groundTruth = df.T.to_dict()

test_users = list(set(groundTruth.keys()).intersection(recs.keys()))
predictionsAndLabels = []
for u in test_users:
    predictionsAndLabels.append((list(recs[u]),groundTruth[u]))
#predictionsAndLabels = predictionsAndLabels[0:100]
predictionsAndLabelsRDD = sc.parallelize(predictionsAndLabels)

metrics = RankingMetrics(predictionsAndLabelsRDD)

metrics.precisionAt(200)
with open('rank30.txt', 'w') as f:
    f.write(str(metrics.precisionAt(200)))