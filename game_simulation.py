#################################
####### Codenames project #######
# By Uri Berger, Tomer Genossar #
########## August 2020 ##########
#################################

'''
File name: game_simulation.py.
Description: This is the main file that generates the clues.
'''

from unharmfulness import distance_unharmfulness,svm_based_unharmfulness,generate_svm_model
from helpfulness import helpfulness1,helpfulness2
import pandas as pd
import numpy as np
import math
import itertools
import os
import scipy as sp
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import copy
import inflect
import csv

# Filenames

board_filename = 'game_words.xlsx'
clue_words_filename = 'clue_nouns.txt'
clues_filename = 'clues.csv'
vectors_filename = 'vectors.kv'

# Model parameters

MAX_BLUE_WORDS_NUM = 2
MIN_BLUE_WORDS_NUM = 1
LAMBDA = 0.5

# Write clues to file flag

WRITE_CLUES_TO_FILE = False
if WRITE_CLUES_TO_FILE:
    # Open file and write column headers
    clues_file = csv.writer(open(clues_filename, 'w', newline=''))
    clues_file.writerow(['Game number','Clue word','Referred blue words'])

# Debug flag

debug = True
def my_print(str):
    if debug:
        print(str)

# Clue generation functions

def generate_game_word_sets(game_number):
    ''' Generate the blue and red word lists, for the given game number. '''
    df = pd.read_excel(board_filename)
    column_list = df.columns.ravel()
    all_words = df['words'].tolist()
    blue_words = [all_words[i].lower().replace(' ','_') for i in range(len(all_words)) if df['colors'][i] == 'blue' and df['game number'][i] == game_number]
    red_words = [all_words[i].lower().replace(' ','_') for i in range(len(all_words)) if df['colors'][i] == 'red' and df['game number'][i] == game_number]
    return blue_words,red_words
        
def generate_clue_words():
    ''' Generate the possible clue words from the clues file. '''
    fp = open(clue_words_filename,'r')
    res = []
    for line in fp:
        line = line.strip()
        res.append(line)
    return res

def generate_clues_by_gensim_similarity(game_words_set, word_vectors):
    ''' Generate the possible clue words according to the gensim similarity:
    Collect all the words that are within the 50 nearest neighbors of one of the words in the given
    game words set, but is not within the 5 nearest neighbors (to avoid clues like 'play' to the
    word 'player'). '''
    res = set()
    for word in game_words_set:
        most_similar_clues = word_vectors.most_similar(positive =[word],topn= 50)
        most_similar_clues = most_similar_clues[5:]
        for x in most_similar_clues:
            res.add(x[0])
    return res

def generate_all_word_vectors():
    ''' Generate the word vectors of the top frequent 500,000 words, from the GoogleNews data set.
    If there's a cache file- load it. Otherwise, generate from scrach, and save it to the cache
    file. '''
    fname = get_tmpfile(vectors_filename)
    if os.path.exists(fname):
        word_vectors = KeyedVectors.load(fname, mmap='r')
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(
            'GoogleNews-vectors-negative300.bin', binary=True, limit=500000
        )
        word_vectors = model.wv
        word_vectors.save(fname)

    return word_vectors
        
def get_best_blue_word_set(helpfulness_func, clue_vec, blue_words_mapping):
    ''' Given a clue word, get the best blue word set that this clue word refers to, according to
    the given helpfulness function. '''
    max_helpfulness = (-1)*math.inf
    best_words_set = None
    
    # Go over all the subsets of blue words in the restricted size
    for blue_words_num in range(MIN_BLUE_WORDS_NUM,MAX_BLUE_WORDS_NUM+1):
        blue_words_sets = list(itertools.combinations(blue_words_mapping.items(), blue_words_num))
        for blue_words_set in blue_words_sets:
            blue_word_vecs = [x[1] for x in blue_words_set]
            cur_helpfulness = helpfulness_func(blue_word_vecs, clue_vec)
            if cur_helpfulness > max_helpfulness:
                max_helpfulness = cur_helpfulness
                best_words_set = [x[0] for x in blue_words_set]
    
    return max_helpfulness,best_words_set

# Keep a cache of all the blue words with 'er' and 'ers' suffix, instead of recalculating it every time
er_cache = None

def is_legal_clue_word(blue_words,clue_word):
    ''' Check if a given clue word is a legal clue for the given blue words. '''
    global er_cache
    
    p = inflect.engine()
    
    # Check if clue is one of the blue words
    if clue_word in blue_words:
        return False
    
    # Check if clue is singular or plural form of one of the blue words
    if p.plural(clue_word) in blue_words:
        return False
    
    # Check if clue with 'er' or 'ers' suffix is one of the blue words
    singular_noun = p.singular_noun(clue_word)
    if singular_noun == False:
        singular_noun = clue_word
    if singular_noun + 'er' in blue_words or singular_noun + 'ers' in blue_words:
        return False
    
    # Check if one of the blue words with 'er' or 'ers' as suffix is equal to the clue
    if er_cache == None:
        er_cache = []
        for blue_word in blue_words:
            singular_noun = p.singular_noun(blue_word)
            if singular_noun == False:
                singular_noun = blue_word
            er_cache.append(singular_noun+'er')
            er_cache.append(singular_noun+'ers')
    if clue_word in er_cache:
        return False
    
    return True

def generate_clue(game_number, helpfulness_func, use_svm_unharmfulness, gensim_clues_list = True):
    ''' Generate the best clue for the given game number. '''
    global er_cache
    er_cache = None
    try:
        blue_words,red_words = generate_game_word_sets(game_number)
        word_vectors = generate_all_word_vectors()
        blue_words_mapping = {x:word_vectors.get_vector(x) for x in blue_words}
        red_words_mapping = {x:word_vectors.get_vector(x) for x in red_words}
        blue_vectors = list(blue_words_mapping.values())
        red_vectors = list(red_words_mapping.values())
        clue_words = generate_clue_words()
        clue_words_mapping = {x:word_vectors.get_vector(x) for x in clue_words}

        '''Find the clue word and blue word set that maximizes the score, i.e.
        lambda*helpfulness + (1-lambda)*unharmfulness '''
        max_score = (-1)*math.inf
        best_clue_word = None
        chosen_blue_words = None

        # Generate the SVM model
        if use_svm_unharmfulness:
            my_print('Generating svm model...')
            svm_model = generate_svm_model(blue_vectors,red_vectors)

        if gensim_clues_list:
            gensim_clue_words = generate_clues_by_gensim_similarity(blue_vectors, word_vectors)
            clue_words = [x for x in gensim_clue_words if x in clue_words]

        # Go over all the clue words, and choose the best one
        for cur_clue_word in clue_words:
            if not is_legal_clue_word(blue_words, cur_clue_word):
                continue

            cur_clue_vec = word_vectors.get_vector(cur_clue_word)

            # Calculate helpfulness
            cur_helpfulness,best_words_set = get_best_blue_word_set(helpfulness_func, cur_clue_vec, blue_words_mapping)

            # Calculate unharmfulness
            if use_svm_unharmfulness:
                cur_unharmfulness = svm_based_unharmfulness(svm_model, np.reshape(cur_clue_vec,(cur_clue_vec.shape[0],1)))
            else:
                cur_unharmfulness = distance_unharmfulness(red_vectors, cur_clue_vec)
            
            cur_score = LAMBDA * cur_helpfulness + (1-LAMBDA)*cur_unharmfulness
            if cur_score > max_score:
                max_score = cur_score
                best_clue_word = cur_clue_word
                chosen_blue_words = best_words_set

        print('Clue word: ' + str(best_clue_word) + ', referred blue words: ' + str(chosen_blue_words))
        print('All the blue words: ' + str(blue_words))
        print('All the red words: ' + str(red_words))
        
        if WRITE_CLUES_TO_FILE:
            chosen_blue_words_str = ';'.join(chosen_blue_words)
            clues_file.writerow([str(game_number),str(best_clue_word),chosen_blue_words_str])
    except KeyError as error:
        print("Word was missing in the dict, the error is:")
        print(error)

for i in range(1,26):
    print("creating clues for game number "+str(i))
    generate_clue(i, helpfulness1,True,False)
    #generate_clue(i, helpfulness1,False,False)