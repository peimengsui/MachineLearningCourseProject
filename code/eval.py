import numpy as np, scipy.sparse as sp
from sklearn.externals import joblib
from sklearn.feature_extraction import FeatureHasher
import random
import MSD_util

# paths to data
_tr = "../data/train_data.txt"
_tv = "../data/test_visible.txt"
_tp = "../data/test_predict.txt"
_tau = 200

# l_rec: list of recommended songs
# sMu: actually listened songs by user
# tau: 200
def AP(l_rec, sMu, tau):
	np = len(sMu)
	nc = 0.0
	mapr_user = 0.0
	for j,s in enumerate(l_rec):
		if j >= tau:
			break
		if s in sMu:
			nc += 1.0
		mapr_user += nc/(j+1)
	mapr_user /= min(np, tau)
	return mapr_user

# l_users: list of users
# l_rec_songs: list of lists, recommended songs for users
# u2s: mapping users to songs
# tau: 20
def mAP(l_users, l_rec_songs, u2s, tau):
	mapr = 0
	n_users = len(l_users)
	for user, l_rec in zip(l_users, l_rec_songs):
		mapr += AP(l_rec, u2s[user], tau)
	return mapr/n_users

def generate_interaction(_tr, _tv, _tp):
	print "Creating user song-interaction lists"
	_, all_songs = MSD_util.get_unique(_tr, users=False, songs=True)
	testv_pairs, testp_pairs = MSD_util.get_user_song_pairs(_tv, _tp)
	return all_songs, testv_pairs, testp_pairs

def generate_features_tr(all_songs, yay_songs_v):
	yay_pairs, nay_pairs, songs_to_catch = [], [], []

	# positive examples
	for song1 in yay_songs_v:
		song_pairs = dict()
		for song2 in yay_songs_v:
			# skip itself to avoid overfitting
			if song1 != song2:
				song_pairs["%s_%s" % (song1, song2)] = 1.
		yay_pairs.append(song_pairs)

	# negative examples: at this point, we don't know what venues will be visited
	nay_songs = all_songs - yay_songs_v
	for song1 in random.sample(nay_songs, len(yay_songs_v)):
		song_pairs = dict()
		for song2 in yay_songs_v:
			song_pairs["%s_%s" % (song1, song2)] = 1.
		nay_pairs.append(song_pairs)

	labels = np.hstack([np.ones(len(yay_pairs)), np.zeros(len(nay_pairs))])
	return labels, yay_pairs, nay_pairs

def generate_features_te(all_songs, yay_songs_p, yay_songs_v):
	yay_pairs, nay_pairs, songs_to_predict = [], [], []

	# positive examples
	for song1 in yay_songs_p:
		song_pairs = dict()
		for song2 in yay_songs_v:
			song_pairs["%s_%s" % (song1, song2)] = 1.
		yay_pairs.append(song_pairs)
		songs_to_predict.append(song1)

	# negative examples
	nay_songs = all_songs - yay_songs_v - yay_songs_p
	for song1 in nay_songs:
		song_pairs = dict()
		for song2 in yay_songs_v:
			song_pairs["%s_%s" % (song1, song2)] = 1.
		nay_pairs.append(song_pairs)
		songs_to_predict.append(song1)

	labels = np.hstack([np.ones(len(yay_pairs)), np.zeros(len(nay_pairs))])
	return labels, yay_pairs, nay_pairs, songs_to_predict

def test(tau):
	all_songs, testv_pairs, testp_pairs = generate_interaction(_tr, _tv, _tp)

	extractor = FeatureHasher(n_features=2**20)
	model = joblib.load('model_log_l2_size20.pkl')

	print "Training"
	for i, (user, yay_songs) in enumerate(testv_pairs.iteritems()):
		if i >= 10000:
			break
		print "Training on user", i, user
		labels, yay_pairs, nay_pairs = generate_features_tr(all_songs, yay_songs)
		yay_features, nay_features = extractor.transform(yay_pairs), extractor.transform(nay_pairs)
		features = sp.vstack([yay_features, nay_features])
		model.partial_fit(features, labels, classes=[0, 1])

	print "Testing"
	l_users, l_rec_songs = [], []
	for i, (user, yay_songs) in enumerate(testp_pairs.iteritems()):
		if i >= 10000:
			break
		print "Testing on user", i, user
		l_users.append(user)
		labels, yay_pairs, nay_pairs, v_to_p = generate_features_te(all_songs, yay_songs, testv_pairs[user])
		yay_features, nay_features = extractor.transform(yay_pairs), extractor.transform(nay_pairs)
		features = sp.vstack([yay_features, nay_features])
		probas = model.predict_proba(features)[:, 1]
		rec_songs = [v_to_p[i] for i in np.argsort(probas)[::-1][:tau]]
		l_users.append(user), l_rec_songs.append(rec_songs)

	print "\nmAP(%d): %f" % (tau, mAP(l_users, l_rec_songs, testp_pairs, tau))

if __name__=="__main__":
	test(_tau)