'''
Created on Apr 25, 2016

@author: minhtran
'''
from configuration import Config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
from utils import load_object, save_object
             
def main():
    # Load configuration
    config = Config()

    # Parse user_list representations
    user_list = []
    user_id_list = []
    with open(config.rep_path, 'r') as data_file:
        lines = data_file.readlines()
        for line in lines:
            user_ = line.split(':')[1].replace('[','').replace(']"}','').split()
            user = [float(u) for u in user_[1:len(user_)]]
            user_list.append(user)
            user_id_list.append(line.split(':')[0].replace('{','').replace('"',''))
    user_list = np.array(user_list)
    user_id_list = np.array(user_id_list)

    # If tsne is already run
    path_user_tsne = os.path.join(os.path.dirname(config.save_path), 'user_tsne')
    if os.path.isfile(path_user_tsne):
        user_tsne = load_object(path_user_tsne)
    else:    
        # Run TSNE
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        user_tsne = model.fit_transform(user_list)    

        # Save TSNE objects
        print "Save user_tsne."
        save_object(user_tsne, 'save/user_tsne')
    
    # Run KMeans clustering
    kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10)
    km = kmeans.fit(user_list)
    
    # Get cluster labels
    labels = km.labels_
    unique_labels = set(labels)

    # Save clustering results
    save_object(user_id_list, 'save/user_ids_km')
    save_object(labels, 'save/labels_km')
        
    # Save the cluster_to_user dict
    cluster_to_user = dict()
    for k in unique_labels:
        class_member_mask = (labels == k)
        class_k = user_id_list[class_member_mask]
        cluster_to_user[k] = class_k
    save_object(cluster_to_user, 'save/cluster_to_user')
    
    # Save the user_to_cluster dict
    user_to_cluster = dict()
    for user, label in zip(user_id_list, labels):
        user_to_cluster[user] = label
    save_object(user_to_cluster, 'save/user_to_cluster')    
    
    # Plot results
    colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = user_tsne[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=3)    

    plt.title('KMeans Clustering')
    plt.show()

if __name__ == '__main__':
    main()
