#################################
####### Codenames project #######
# By Uri Berger, Tomer Genossar #
########## August 2020 ##########
#################################

'''
File name: helpfulness.py.
Description: This file contains the different helpfulness functions.
'''

import scipy as sp
import numpy as np
import math
from copy import deepcopy

def average_vec(vec_list):
    ''' Calculates the average vector, given a vector list. '''
    sum_vec = deepcopy(vec_list[0])
    for i in range(1,len(vec_list)):
        sum_vec += vec_list[i]
    return sum_vec / len(vec_list)

def helpfulness1(blue_word_vecs, clue_word_vec):
    ''' Helpfulness function 1: measures the maximal distance from a blue word, normalized by the
    number of blue words this clue refers to. '''
    n = len(blue_word_vecs)
    return ((-1) * max([sp.spatial.distance.cosine(clue_word_vec, x) for x in blue_word_vecs]))/n
    
def helpfulness2(blue_word_vecs, clue_word_vec):
    ''' Helpfulness function 2: measures the variance of the distance from the blue words. '''
    blue_words_average_vec = average_vec(blue_word_vecs)
    
    var_list = [(np.linalg.norm(clue_word_vec-x)-np.linalg.norm(clue_word_vec-blue_words_average_vec))**2 for x in blue_word_vecs]
    return (-1)*sum(var_list)