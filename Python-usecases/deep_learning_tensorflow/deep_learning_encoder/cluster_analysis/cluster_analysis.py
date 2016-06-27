'''
Created on Apr 28, 2016

@author: minhtran
'''
from configuration import Config
from itertools import islice
import json
from utils import load_object
import numpy as np
from dtw import dtw
import heapq
import random

config = Config()

# Read the input file to get behaviours of users
def get_user_behave():
    # The user_to_behave dict
    user_to_behave = dict()
    
    # Read data chunk by chunk
    with open(config.data_path, 'r') as data_file:
        while True:
            # Get one batch at a time
            exit = True
            lines_gen = islice(data_file, config.batch_size)
            for line in lines_gen:
                user_behave = json.loads(line).values()[0]
                
                # Remove those users who only have 1 event
                if (len(user_behave) <= config.num_min_event):
                    continue

                if (len(user_behave) < config.num_event + 1):
                    zero_pad = [[0 for _ in range(len(user_behave[0]))] 
                                for _ in range(config.num_event-len(user_behave)+1)]
                    user_behave = user_behave + zero_pad
                elif (len(user_behave) > config.num_event + 1):
                    user_behave = user_behave[(len(user_behave)-config.num_event-1) 
                                              : len(user_behave)]
                
                # Add to dict                       
                user_id = json.loads(line).keys()[0]
                user_to_behave[user_id] = np.array(user_behave)[:, list([1,2,3,7,8,9,10,11,12,13])]
                exit = False
            
            # Batch is empty then exit
            if exit: break
    return user_to_behave

# Get list of user_ids from within a cluster
def get_user_of_cluster(cluster_id):
    cluster_to_user = load_object("save/cluster_to_user")
    return cluster_to_user[cluster_id]

# Get top features that impact on a cluster
def get_top_feature(cluster_id, user_to_behave):
    user_list = get_user_of_cluster(cluster_id) 
    behave_batch = []
    for user in user_list:
        user_behave = user_to_behave[user]
        behave_batch.append(user_behave)
    batch_to_arr = np.asarray(behave_batch)
    num_feature = len(config.feature_desc)

    aver_dist = []
    for feature in range(num_feature):
        index_list = [f for f in range(num_feature) if f != feature]
        new_batch = batch_to_arr[:, :, index_list]
        aver_dist.append(get_aver_dist_1_cluster(new_batch))
    
    var_imp = np.array((aver_dist - np.min(aver_dist)) / (np.max(aver_dist) - np.min(aver_dist))) 
    print "Feature importance..."
    print var_imp
    
    return heapq.nlargest(config.num_top_feature, range(len(var_imp)), var_imp.take)

# Get average distance for 1 cluster        
def get_aver_dist_1_cluster(batch):
    aver_dist = 0.0
    num_user = range(len(batch))
    index = []
    for _ in range(200):
        choice = random.choice(num_user)
        index.append(choice)
        num_user.remove(choice)

    for i in index[0:100]:
        for j in index[100:200]:
            dist = dtw(batch[i], batch[j], dist=custom_dist)
            aver_dist = aver_dist + dist[0]
    return float(aver_dist) / (len(index) * len(index) / 4)
    
# Function to compute the distance between 2 feature vectors
def custom_dist(feature_vect1, feature_vect2):
    dist = 0.0
    for f1, f2 in zip(feature_vect1, feature_vect2):
        if f1 != f2: dist = dist + 1
    return dist 

# Get average distance for 2 clusters    
def get_aver_dist_2_cluster(batch1, batch2):
    aver_dist = 0.0
    num_user1 = range(len(batch1))
    index1 = []
    for _ in range(100):
        choice = random.choice(num_user1)
        index1.append(choice)
        num_user1.remove(choice)    

    num_user2 = range(len(batch2))
    index2 = []
    for _ in range(100):
        choice = random.choice(num_user2)
        index2.append(choice)
        num_user2.remove(choice)       
    
    for i in index1:
        for j in index2:
            dist = dtw(batch1[i], batch2[j], dist=custom_dist)
            aver_dist = aver_dist + dist[0]
    return float(aver_dist) / (len(index1) * len(index2))

# Get top features that separate 2 clusters
def get_distinguish_feature(cluster_id1, cluster_id2, user_to_behave):
    # Process for cluster1
    user_list1 = get_user_of_cluster(cluster_id1) 
    behave_batch1 = []
    for user in user_list1:
        user_behave = user_to_behave[user]
        behave_batch1.append(user_behave)
    batch_to_arr1 = np.asarray(behave_batch1)
    
    # Process for cluster2
    user_list2 = get_user_of_cluster(cluster_id2) 
    behave_batch2 = []
    for user in user_list2:
        user_behave = user_to_behave[user]
        behave_batch2.append(user_behave)
    batch_to_arr2 = np.asarray(behave_batch2)    
    
    num_feature = len(config.feature_desc)
    
    aver_dist = []
    for feature in range(num_feature):
        index_list = [f for f in range(num_feature) if f != feature]
        new_batch1 = batch_to_arr1[:, :, index_list]
        new_batch2 = batch_to_arr2[:, :, index_list]
        aver_dist.append(get_aver_dist_2_cluster(new_batch1, new_batch2))
    
    var_imp = 1 - np.array((aver_dist - np.min(aver_dist)) / (np.max(aver_dist) - np.min(aver_dist))) 
    print "Feature importance..."
    print var_imp
    
    return heapq.nlargest(config.num_top_feature, range(len(var_imp)), var_imp.take)
    
def main():
    # Set cluster ids for analysis
    cluster_id1 = 3
    cluster_id2 = 1
    
    # Get user behaviours
    user_to_behave = get_user_behave()

    # Compute top features for cluster_id1
    top_features = get_top_feature(cluster_id1, user_to_behave)
    print "Top %d features that make cluster %d distinct are:" % (config.num_top_feature, cluster_id1)
    print top_features
    
    # Compute top distinguished features for cluster_id1 and cluster_id2
    dis_features = get_distinguish_feature(cluster_id1, cluster_id2, user_to_behave)
    print "Top %d features that separate cluster %d and cluster %d are:" % (config.num_top_feature, 
                                                                      cluster_id1, cluster_id2)
    print dis_features
        
if __name__ == "__main__":
    main()