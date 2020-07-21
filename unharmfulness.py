import numpy as np
from sklearn import svm

def distance_unharmfulness(rival_team_word_vecs_list, clue_word_vec):
    return sum([np.linalg.norm(clue_word_vec-x) for x in rival_team_word_vecs_list])

def generate_svm_model(pos_vecs_list, neg_vecs_list):
    all_samples = np.zeros((len(pos_vecs_list)+len(neg_vecs_list),pos_vecs_list[0].shape[0]))
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

def svm_based_unharmfulness(svm_model, clue_word_vec):
    return svm_model.decision_function(clue_word_vec.transpose())[0]