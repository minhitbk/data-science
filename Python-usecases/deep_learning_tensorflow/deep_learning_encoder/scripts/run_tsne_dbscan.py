'''
Created on Apr 25, 2016

@author: minhtran
'''
from configuration import Config
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
from utils import load_object, save_object

def main():
    # Load configuration
    config = Config()

    # Parse user_list representations
    user_list, user_ids = [], []
    with open(config.rep_path, "r") as data_file:
        lines = data_file.readlines()
        for line in lines:
            user_ = line.split(":")[1].replace("[", "").replace(']"}', "").split()
            id_ = line.split(":")[0].replace("{", "").replace('"', "")
            user = [float(u) for u in user_[1:len(user_)]]
            user_list.append(user)
            user_ids.append(id_)  
    user_list = np.array(user_list)

    # If tsne is already run
    path_user_tsne = os.path.join(os.path.dirname(config.save_path), "user_tsne")
    if os.path.isfile(path_user_tsne):
        user_tsne = load_object(path_user_tsne)
    else:    
        # Run TSNE
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        user_tsne = model.fit_transform(user_list)    

        # Save TSNE objects
        print "Save user_tsne."
        save_object(user_tsne, "save/user_tsne")
    
    # Run DBSCAN
    db = DBSCAN(eps=3, min_samples=50, algorithm="brute").fit(user_tsne)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Save clustering results
    save_object(user_ids, "save/user_ids_db")
    save_object(labels, "save/labels_db")
    
    # Drawing clustering
    unique_labels = set(labels)
    colors = plt.get_cmap("Spectral")(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1: continue  
        class_member_mask = (labels == k)    
        xy = user_tsne[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=col, markeredgecolor="k", markersize=6)    
        xy = user_tsne[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=col, markeredgecolor="k", markersize=3)       
    
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()

if __name__ == "__main__":
    main()
