import scipy as sp
import numpy as np
import math
from copy import deepcopy

def average_vec(vec_list):
    sum_vec = deepcopy(vec_list[0])
    for i in range(1,len(vec_list)):
        sum_vec += vec_list[i]
    return sum_vec / len(vec_list)

def helpfulness1(blue_word_vecs, clue_word_vec):
    n = len(blue_word_vecs)
    
    blue_words_average_vec = average_vec(blue_word_vecs)
    
    return (-1) * n * sp.spatial.distance.cosine(clue_word_vec, blue_words_average_vec)
    #return (-1) * n * np.linalg.norm(clue_word_vec-blue_words_average_vec)
    
def helpfulness2(blue_word_vecs, clue_word_vec):
    blue_words_average_vec = average_vec(blue_word_vecs)
    
    var_list = [(np.linalg.norm(clue_word_vec-x)-np.linalg.norm(clue_word_vec-blue_words_average_vec))**2 for x in blue_word_vecs]
    return (-1)*sum(var_list)