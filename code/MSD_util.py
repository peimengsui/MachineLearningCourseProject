import random, time, math
import sys, os
import numpy as np
from collections import defaultdict

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