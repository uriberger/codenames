import numpy as np
from sklearn import svm

def distance_unharmfulness(rival_team_word_vecs_list, clue_word_vec):
    return sum([np.abs(clue_word_vec-x) for x in rival_team_word_vecs_list])

def generate_separating_plane_weight_vec(pos_vecs_list, neg_vecs_list):
    all_samples = np.zeros((len(pos_vecs_list)+len(neg_vecs_list),pos_vecs_list[0].shape))
    loc = 0
    for pos_vec in pos_vecs_list:
        all_samples[[loc],:] = pos_vec.transpose()
        loc += 1
    for neg_vec in neg_vecs_list:
        all_samples[[loc],:] = neg_vec.transpose()
        loc += 1
    labels = [1]*len(pos_vecs_list) + [-1]*len(neg_vecs_list)
    
    clf = svm.SVC()
    clf.fit(all_samples,labels)
    
    return clf

def svm_based_unharmfulness(own_team_word_vecs_list, rival_team_word_vecs_list, clue_word_vec):
    clf = generate_separating_plane_weight_vec(own_team_word_vecs_list, rival_team_word_vecs_list)
    return clf.decision_function(clue_word_vec.transpose())[0]