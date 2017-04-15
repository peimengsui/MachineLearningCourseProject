# http://rnowling.github.io/data/science/2016/10/20/lr-hashing-recsys.html
# https://github.com/rnowling/rec-sys-experiments

import numpy as np, scipy.sparse as sp
import random, sys
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.externals import joblib
import MSD_util
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

# paths to data
f_triplets_tr = "../data/train_data.txt"
# f_triplets_vv = "../data/valid_visible.txt"
# f_triplets_vp = "../data/valid_predict.txt"
f_triplets_va = "../data/valid_data.txt"

def generate_interaction(_tr, _vv):
	print "Creating user song-interaction lists"
	_, all_songs = MSD_util.get_unique(_tr, users=False, songs=True)
	train_pairs, valid_pairs = MSD_util.get_user_song_pairs(_tr, _vv)
	return all_songs, train_pairs, valid_pairs

def generate_features(all_songs, listened_songs):
	listened_pairs, unlistened_pairs = [], []

	# positive examples
	for song1 in listened_songs:
		song_pairs = dict()
		for song2 in listened_songs:
			# skip itself to avoid overfitting
			if song1 != song2:
				song_pairs["%s_%s" % (song1, song2)] = 1.
		listened_pairs.append(song_pairs)

	# negative examples
	unlistened_songs = all_songs - listened_songs
	for song1 in random.sample(unlistened_songs, len(listened_songs)):
		song_pairs = dict()
		for song2 in listened_songs:
			song_pairs["%s_%s" % (song1, song2)] = 1.
		unlistened_pairs.append(song_pairs)

	labels = np.hstack([np.ones(len(listened_pairs)), np.zeros(len(unlistened_pairs))])
	return labels, listened_pairs, unlistened_pairs

def train_and_score(_tr, _va, model_size):
	all_songs, train_pairs, valid_pairs = generate_interaction(_tr, _va)
	# extractors, models = dict(), dict()

	print "Creating model"
	extractor = FeatureHasher(n_features=2**model_size)
	model = SGDClassifier(loss="log", penalty="L2")

	print "Training"
	for i, (user, listened_songs) in enumerate(train_pairs.iteritems()):
		print "Training on user", i, user
		labels, listened_pairs, unlistened_pairs = generate_features(all_songs, listened_songs)
		listened_features = extractor.transform(listened_pairs)
		unlistend_features = extractor.transform(unlistened_pairs)
		features = sp.vstack([listened_features, unlistend_features])
		model.partial_fit(features, labels, classes=[0, 1])

	joblib.dump(model, 'model_log_l2_size%d.pkl' % model_size)
	model = joblib.load('model_log_l2_size%d.pkl' % model_size) 

	print "Testing"
	all_labels, all_preds, all_probas = [], [], []
	for i, (user, listened_songs) in enumerate(valid_pairs.iteritems()):
		print "Testing on user", i, user
		labels, listened_pairs, unlistened_pairs = generate_features(all_songs, listened_songs)
		all_labels.extend(labels)
		listened_features = extractor.transform(listened_pairs)
		unlistend_features = extractor.transform(unlistened_pairs)
		features = sp.vstack([listened_features, unlistend_features])
		preds = model.predict(features)
		probas = model.predict_proba(features)
		all_preds.extend(preds), all_probas.extend(probas[:, 1])

	print "Scoring"
	roc_auc = roc_auc_score(all_labels, all_probas)
	cm = confusion_matrix(all_labels, all_preds)
	print "Model size", model_size, "AUC", roc_auc
	print cm

	fpr, tpr, _ = roc_curve(all_labels, all_probas)
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
		lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic for model size %d' % model_size)
	plt.legend(loc="lower right")
	plt.show()

def main():
	# model size by number of bits
	model_size = int(sys.argv[1])
	train_and_score(f_triplets_tr, f_triplets_va, model_size)

if __name__=="__main__":
	main()