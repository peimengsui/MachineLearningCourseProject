import random, time, math, cPickle
import sys, os
import numpy as np, pandas as pd
from collections import defaultdict, Counter

def get_user_song_pairs(_tr, _va):
    train_pairs, valid_pairs = defaultdict(set), defaultdict(set)
    with open(_tr, "r") as f:
        for line in f:
            user, song, _ = line.strip().split("\t")
            train_pairs[user].add(song)
    with open(_va, "r") as f:
        for line in f:
            user, song, _ = line.strip().split("\t")
            valid_pairs[user].add(song)
    return train_pairs, valid_pairs

def song_to_count(if_str, binary=True):
    stc  = dict()
    with open(if_str, "r") as f:
        for line in f:
            if binary:
                _, song, _ = line.strip().split('\t')
                if song in stc:
                    stc[song] += 1
                else:
                    stc[song] = 1
            else:
                _, song, freq = line.strip().split('\t')
                if song in stc:
                    stc[song] += int(freq)
                else:
                    stc[song] = int(freq)
    return stc

def sort_dict_dec(d):
    return sorted(d.keys(), key=lambda s: d[s], reverse=True)

def clean_songs():
    stc = song_to_count("../data/train_triplets.txt", binary=True)
    songs_to_keep = [song for song in stc if stc[song]>=10]
    del stc
    with open("../data/songs_to_keep.pkl", "wb") as output_file:
        cPickle.dump(songs_to_keep, output_file)
    with open("../data/songs_to_keep.pkl", "rb") as input_file:
        songs_to_keep = cPickle.load(input_file)
    train = pd.read_table("../data/train_data.txt", names=['user', 'song', 'count'], dtype={'count': str})
    train = train[train['song'].isin(songs_to_keep)]
    with open("../data/train_data_.txt", "w") as f:
        f.write("\n".join(["\t".join(v) for v in train.values]))
    print "train data processed"
    valid_v = pd.read_table("../data/valid_visible.txt", names=['user', 'song', 'count'], dtype={'count': str})
    valid_v = valid_v[valid_v['song'].isin(songs_to_keep)]
    with open("../data/valid_visible_.txt", "w") as f:
        f.write("\n".join(["\t".join(v) for v in valid_v.values]))
    print "valid visible processed"
    valid_p = pd.read_table("../data/valid_predict.txt", names=['user', 'song', 'count'], dtype={'count': str})
    valid_p = valid_p[valid_p['song'].isin(songs_to_keep)]
    with open("../data/valid_predict_.txt", "w") as f:
        f.write("\n".join(["\t".join(v) for v in valid_p.values]))
    print "valid predict processed"
    test = pd.read_table("../data/test_data.txt", names=['user', 'song', 'count'], dtype={'count': str})
    test = test[test['song'].isin(songs_to_keep)]
    with open("../data/test_data_.txt", "w") as f:
        f.write("\n".join(["\t".join(v) for v in test.values]))
    print "test data processed"

def clean_users():
    with open("../data/songs_to_keep.pkl", "rb") as input_file:
        songs_to_keep = cPickle.load(input_file)
    train = pd.read_table("../data/train_triplets.txt", names=['user', 'song', 'count'], dtype={'count': str})
    train = train[train['song'].isin(songs_to_keep)]
    users_to_keep = [u[0] for u in Counter(list(train['user'])).most_common() if u[1]>=10]
    print len(users_to_keep), "users to keep"
    train = pd.read_table("../data/train_data_.txt", names=['user', 'song', 'count'], dtype={'count': str})
    train = train[train['user'].isin(users_to_keep)]
    with open("../data/train_data_.txt", "w") as f:
        f.write("\n".join(["\t".join(v) for v in train.values]))
    print "train data processed"
    valid_v = pd.read_table("../data/valid_visible_.txt", names=['user', 'song', 'count'], dtype={'count': str})
    valid_v = valid_v[valid_v['user'].isin(users_to_keep)]
    with open("../data/valid_visible_.txt", "w") as f:
        f.write("\n".join(["\t".join(v) for v in valid_v.values]))
    print "valid visible processed"
    valid_p = pd.read_table("../data/valid_predict_.txt", names=['user', 'song', 'count'], dtype={'count': str})
    valid_p = valid_p[valid_p['user'].isin(users_to_keep)]
    with open("../data/valid_predict_.txt", "w") as f:
        f.write("\n".join(["\t".join(v) for v in valid_p.values]))
    print "valid predict processed"
    test = pd.read_table("../data/test_data_.txt", names=['user', 'song', 'count'], dtype={'count': str})
    test = test[test['user'].isin(users_to_keep)]
    with open("../data/test_data_.txt", "w") as f:
        f.write("\n".join(["\t".join(v) for v in test.values]))
    print "test data processed"

def get_unique(if_str, users, songs):
    u, s = set(), set()
    with open(if_str, "r") as f:
        for line in f:
            if users and songs:
                user, song, _ = line.strip().split('\t')
                if user not in u:
                    u.add(user)
                if song not in s:
                    s.add(song)
            elif users:
                user, _, _ = line.strip().split('\t')
                if user not in u:
                    u.add(user)
            elif songs:
                _, song, _ = line.strip().split('\t')
                if song not in s:
                    s.add(song)
    return u, s 

def song_to_users(if_str, u2i):
    stu = dict()
    with open(if_str,"r") as f:
        for line in f:
            user, song, _ = line.strip().split('\t')
            if song in stu:
                stu[song].add(u2i[user])
            else:
                stu[song] = set([u2i[user]])
    return stu

def user_to_songs(if_str):
    uts=dict()
    with open(if_str,"r") as f:
        for line in f:
            user,song,_=line.strip().split('\t')
            if user in uts:
                uts[user].add(song)
            else:
                uts[user]=set([song])
    return uts

def unique_users(if_str):
    u=set()
    with open(if_str,"r") as f:
        for line in f:
            user,_,_=line.strip().split('\t')
            if user not in u:
                u.add(user)
    return u 

def save_recommendations(r,songs_file,ofile):
    print "Loading song indices from " + songs_file
    s2i=song_to_idx(songs_file)
    print "Saving recommendations"
    f=open(ofile,"w")
    for r_songs in r:
        indices=map(lambda s: s2i[s],r_songs)
        f.write(" ".join(indices)+"\n")
    f.close()
    print "Ok."