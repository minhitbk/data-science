'''
Created on May 16, 2016

@author: minhtran
'''
import tensorflow as tf
from encoder import AutoEncoder
from configuration import Config
import numpy as np
from itertools import islice
import json
from utils import load_object, save_object
import os
from tensorflow.python.ops import array_ops

# Load configuration
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
                
                # Remove those users who have less than num_min_event
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
    
    # Save user_to_behave object
    save_object(user_to_behave, os.path.join(os.path.dirname(config.save_path), 
                                             'user_to_behave.pkl'))
    return user_to_behave

def get_embeddings(sess):
    embeddings = dict()
    for num_cat_value in np.unique(config.feature_desc):
        embed_prefix_name = 'embeddings' + str(num_cat_value)
        embed_var = [v for v in tf.all_variables() if v.name.startswith(embed_prefix_name)][0]
        embeddings[num_cat_value] = sess.run(embed_var)
    return embeddings
    
def get_user_embedding(sess, user_id):
    # If user_to_behave is already run
    path_user_to_behave = os.path.join(os.path.dirname(config.save_path), 'user_to_behave.pkl')
    if os.path.isfile(path_user_to_behave):
        user_to_behave = load_object(path_user_to_behave)
    else:    
        user_to_behave = get_user_behave()
    
    # Get embeddings 
    embeddings = get_embeddings(sess)
    
    # Get embeddings for each event
    embed_events = []
    for event in user_to_behave[user_id]:
        embed_event = []
        for feature in range(len(event)):
            num_cat_value = config.feature_desc[feature]
            if num_cat_value == 1:
                embed_event.append(list(event[feature]))
            else:
                embed_event.append((embeddings[num_cat_value][event[feature]]))
        embed_events.append(embed_event)
            
    return embed_events
    
def feature_importance(sess, user_id, matrix_i, matrix_j, matrix_f, matrix_o, 
                       bias_i, bias_j, bias_f, bias_o):
    user_embedding = get_user_embedding(sess, user_id)
    end_index = 0
    gates_i, gates_j, gates_f, gates_o = [], [], [], []    
    for feature in range(len(config.feature_desc)):
        start_index = end_index
        end_index = start_index + config.feature_desc[feature]
        gate_i, gate_j, gate_f, gate_o = 0, 0, 0, 0
        for event in user_embedding:
            gate_i += np.sum(sess.run(tf.matmul(tf.reshape(event[feature], [1, -1]), 
                                                matrix_i[start_index:end_index]) + 
                                      tf.reshape(bias_i, [1, -1])))
            gate_j += np.sum(sess.run(tf.matmul(tf.reshape(event[feature], [1, -1]), 
                                                matrix_j[start_index:end_index]) + 
                                      tf.reshape(bias_j, [1, -1])))
            gate_f += np.sum(sess.run(tf.matmul(tf.reshape(event[feature], [1, -1]), 
                                                matrix_f[start_index:end_index]) + 
                                      tf.reshape(bias_f, [1, -1])))
            gate_o += np.sum(sess.run(tf.matmul(tf.reshape(event[feature], [1, -1]), 
                                                matrix_o[start_index:end_index]) + 
                                      tf.reshape(bias_o, [1, -1])))
                     
        gates_i.append(gate_i/len(user_embedding))
        gates_j.append(gate_j/len(user_embedding))
        gates_f.append(gate_f/len(user_embedding))
        gates_o.append(gate_o/len(user_embedding))
    return gates_i, gates_j, gates_f, gates_o

def cluster_feature_analysis(sess, user_ids):
    # Get trained parameters
    lstm_vars = [v for v in tf.all_variables() if v.name.startswith('lstm')]
    matrix_var = sess.run(lstm_vars[0])
    bias_var = sess.run(lstm_vars[1])
    
    # Split the gates
    matrix_i, matrix_j, matrix_f, matrix_o = sess.run(array_ops.split(1, 4, matrix_var))
    bias_i, bias_j, bias_f, bias_o = sess.run(array_ops.split(0, 4, bias_var))
    
    dict_i, dict_j, dict_f, dict_o = dict(), dict(), dict(), dict()
    for feature in range(len(config.feature_desc)):
        dict_i[feature] = []
        dict_j[feature] = []
        dict_f[feature] = []
        dict_o[feature] = []
    for user_id in user_ids:
        print user_id
        gates_i, gates_j, gates_f, gates_o = feature_importance(sess, user_id, matrix_i, 
                                                                matrix_j, matrix_f, matrix_o, 
                                                                bias_i, bias_j, bias_f, bias_o)
        for feature in range(len(config.feature_desc)):
            dict_i[feature].append(gates_i[feature])
            dict_j[feature].append(gates_j[feature])
            dict_f[feature].append(gates_f[feature])
            dict_o[feature].append(gates_o[feature])                        
    return dict_i, dict_j, dict_f, dict_o

# The main function                            
def main(_):   
    # Rebuild the graph
    def_graph = tf.Graph().as_default()
    auto_encoder = AutoEncoder(config)
    auto_encoder.build_encoder(config.feature_desc)
    
    # Create session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    # Load the auto encoding model
    saver = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state('save')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
  
    # Analyse DBScan results on t-sne
    user_ids_db = np.array(load_object('save/user_ids_db'))
    labels_db = load_object('save/labels_db')
    
    user_ids1 = user_ids_db[(labels_db==2)][0:30]
    user_ids2 = user_ids_db[(labels_db==6)][0:30]
     
    cluster1 = cluster_feature_analysis(sess, user_ids1)
    cluster2 = cluster_feature_analysis(sess, user_ids2)
    
    save_object(cluster1, 'save/cluster1_db')
    save_object(cluster2, 'save/cluster2_db')
    
    # Analyse K-means results on reps
    user_ids_km = np.array(load_object('save/user_ids_km'))
    labels_km = load_object('save/labels_km')
    
    user_ids1 = user_ids_km[(labels_km==2)][0:30]
    user_ids2 = user_ids_km[(labels_km==6)][0:30]
     
    cluster1 = cluster_feature_analysis(sess, user_ids1)
    cluster2 = cluster_feature_analysis(sess, user_ids2)    

    save_object(cluster1, 'save/cluster1_km')
    save_object(cluster2, 'save/cluster2_km')

if __name__ == "__main__":
    tf.app.run()