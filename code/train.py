import sys
import MSD_util,MSD_rec

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

recommender.Valid(_tau, u2s_vv.keys(), u2s_vv, u2s_vp, _n_batch)
# recs = recommender.RecommendToUsers(users_v[user_min:user_max], u2s_v)
# MSD_util.save_recommendations(recs, "../data/kaggle_songs.txt", osfile)