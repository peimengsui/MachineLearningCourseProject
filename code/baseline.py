import sys
import MSD_util, MSD_rec

# paths to data
f_triplets_tr = "../data/train_data.txt"
f_triplets_vv = "../data/valid_visible.txt"
f_triplets_vp = "../data/valid_predict.txt"

# parameters
_tau = 500

print 'default ordering by popularity'
sys.stdout.flush()
songs_ordered = MSD_util.sort_dict_dec(MSD_util.song_to_count(f_triplets_tr, binary=False))

print 'user to songs on %s'%f_triplets_vv
u2s_vv = MSD_util.user_to_songs(f_triplets_vv)
print 'user to songs on %s'%f_triplets_vp
u2s_vp = MSD_util.user_to_songs(f_triplets_vp)

# recommend top N most popular songs (extremely unpersonalized :|)
all_recs = []
for u in u2s_vv:
    recs_500 = set(songs_ordered[:500])-u2s_vv[u]
    recs4u = list(recs_500)
    if len(recs4u)<500:
        n_more = 500-len(recs4u)
        recs4u += songs_ordered[500:500+n_more]
    all_recs.append(recs4u)

map_all = MSD_rec.mAP(u2s_vv.keys(), all_recs, u2s_vp, _tau)
print
print "mAP for %d users in the validation set: %f"%(len(u2s_vv), map_all)
# binary: 0.004516
# frequency-based: 0.004643