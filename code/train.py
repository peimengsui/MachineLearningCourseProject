import sys
import MSD_util,MSD_rec
#user_min beginning user index
#user_max ending user index
# mode 1:user based mode 2:song based
user_min,user_max,mode=sys.argv[1:]
user_min=int(user_min)
user_max=int(user_max)
mode=int(mode)
print "user_min: %d , user_max: %d"%(user_min,user_max)
# paths to data
f_triplets_tr = "../data/train_data.txt"
f_triplets_vv = "../data/valid_visible.txt"
f_triplets_vp = "../data/valid_predict.txt"

# parameters
_A = 0.5
_Q = 1
_tau = 500
_Gamma = [1.0]
_n_batch = 10

if mode==1:
	print 'default ordering by popularity'
	sys.stdout.flush()
	songs_ordered = MSD_util.sort_dict_dec(MSD_util.song_to_count(f_triplets_tr))
	print 'user to songs on %s'%f_triplets_tr
	u2s_tr = MSD_util.user_to_songs(f_triplets_tr)
	print 'user to songs on %s'%f_triplets_vv
	u2s_vv = MSD_util.user_to_songs(f_triplets_vv)
	print 'user to songs on %s'%f_triplets_vp
	u2s_vp = MSD_util.user_to_songs(f_triplets_vp)
	print 'Creating predictor...'
	predictor = MSD_rec.PredSU(u2s_tr, _A, _Q)
	print 'Creating recommender..'
	recommender = MSD_rec.Reco(songs_ordered, _tau, _Gamma)
	recommender.Add(predictor)

	recommender.Valid(_tau, u2s_vv.keys()[user_min:user_max], u2s_vv, u2s_vp, _n_batch)
if mode==2:
	print 'default ordering by popularity'
	sys.stdout.flush()
	songs_ordered = MSD_util.sort_dict_dec(MSD_util.song_to_count(f_triplets_tr))

	print  "loading unique users indexes"
	uu = MSD_util.unique_users(f_triplets_tr)
	u2i = {u:i for i,u in enumerate(uu)}

	print 'song to users on %s'%f_triplets_tr
	s2u_tr = MSD_util.song_to_users(f_triplets_tr, u2i)

	del u2i

	print 'user to songs on %s'%f_triplets_vv
	u2s_vv = MSD_util.user_to_songs(f_triplets_vv)
	print 'user to songs on %s'%f_triplets_vp
	u2s_vp = MSD_util.user_to_songs(f_triplets_vp)

	print 'Creating predictor...'
	predictor = MSD_rec.PredSI(s2u_tr, _A, _Q)

	print 'Creating recommender..'
	recommender = MSD_rec.Reco(songs_ordered, _tau, _Gamma)
	recommender.Add(predictor)

	recommender.Valid(_tau, u2s_vv.keys()[user_min:user_max], u2s_vv, u2s_vp, _n_batch)
# recs = recommender.RecommendToUsers(users_v[user_min:user_max], u2s_v)
# MSD_util.save_recommendations(recs, "../data/kaggle_songs.txt", osfile)