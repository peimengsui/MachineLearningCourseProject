import random, time, math
import sys, os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import scipy.io

def song_to_count(if_str):
    stc  = dict()
    with open(if_str,"r") as f:
        for line in f:
            _,song,_ = line.strip().split('\t')
            if song in stc:
                stc[song] += 1
            else:
                stc[song] = 1
    return stc

def sort_dict_dec(d):
    return sorted(d.keys(),key=lambda s:d[s],reverse=True)

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
    return sorted(u), sorted(s) 

def load_data(if_str, users, songs):
    users_dict = {u:i for i,u in enumerate(users)}
    songs_dict = {u:i for i,u in enumerate(songs)}
    nrows, ncols = len(users), len(songs)
    print "matrix dim: %d X %d"%(nrows, ncols)
    row, col, data = [], [], []
    with open(if_str, "r") as f:
        for line in f:
            user, song, val = line.strip().split('\t')
            row.append(users_dict[user])
            col.append(songs_dict[song])
            data.append(int(val))
    row, col, data = np.array(row), np.array(col), np.array(data)
    mat = csr_matrix((data, (row, col)), shape=(nrows, ncols))
    train_valid, test = train_test_split(mat, test_size=0.2, random_state=42)
    train, valid = train_test_split(train_valid, test_size=0.25, random_state=41)
    scipy.io.savemat('../data/train.mat', {'train': train})
    scipy.io.savemat('../data/valid.mat', {'valid': valid})
    scipy.io.savemat('../data/test.mat', {'test': test})
    return train, valid, test

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