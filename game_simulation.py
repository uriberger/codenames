from unharmfulness import distance_unharmfulness,svm_based_unharmfulness,generate_svm_model
from helpfulness import helpfulness1,helpfulness2
import pandas as pd
import numpy as np
import spacy
import math
import itertools
import en_core_web_sm

MAX_BLUE_WORDS_NUM = 4
MIN_BLUE_WORDS_NUM = 4
LAMBDA = 0.5

debug = True
def my_print(str):
    if debug:
        print(str)

board_filename = 'game_words.xlsx'
all_words_filename = 'all_words.txt'
my_print('Loading nlp model...')
nlp = en_core_web_sm.load()

def generate_game_word_sets(game_number):
    df = pd.read_excel(board_filename)
    column_list = df.columns.ravel()
    all_words = df['words'].tolist()
    blue_words = [all_words[i] for i in range(len(all_words)) if df['colors'][i] == 'blue' and df['game number'][i] == game_number]
    red_words = [all_words[i] for i in range(len(all_words)) if df['colors'][i] == 'red' and df['game number'][i] == game_number]
    return blue_words,red_words

def generate_game_tokens(game_number):
    my_print('Generating word sets...')
    words_set = generate_game_word_sets(game_number)
    
    my_print('Generating blue tokens...')
    blue_tokens = []
    for word in words_set[0]:
        blue_tokens.append(nlp(word))
    
    my_print('Generating red tokens...')
    red_tokens = []
    for word in words_set[1]:
        red_tokens.append(nlp(word))
        
    return blue_tokens,red_tokens
        
def generate_all_words_set():
    fp = open(all_words_filename,'r')
    res = []
    for line in fp:
        line = line.strip()
        res.append(line)
    return res
        
def generate_all_words_tokens():
    my_print('Generating all words set...')
    words_set = generate_all_words_set()
    
    my_print('Generating all words tokens...')
    res = []
    for word in words_set:
        res.append(nlp(word))
        
    return res

def get_best_blue_word_set(helpfulness_func, clue_word_token, blue_word_tokens):
    # Given a clue word, get the best blue word set that this clue word refers to
    max_helpfulness = (-1)*math.inf
    best_word_token_set = None
    
    # Go over all the subsets of blue words in the restricted size
    for blue_words_num in range(MIN_BLUE_WORDS_NUM,MAX_BLUE_WORDS_NUM+1):
        blue_word_token_sets = list(itertools.combinations(blue_word_tokens, blue_words_num))
        for blue_word_token_set in blue_word_token_sets:
            blue_word_vecs = [x.vector for x in blue_word_token_set]
            cur_helpfulness = helpfulness_func(blue_word_vecs, clue_word_token.vector)
            if cur_helpfulness > max_helpfulness:
                max_helpfulness = cur_helpfulness
                best_word_token_set = blue_word_token_set
    
    return max_helpfulness,best_word_token_set

def generate_clue(game_number, helpfulness_func, use_svm_unharmfulness):
    blue_tokens,red_tokens = generate_game_tokens(game_number)
    blue_vectors = [x.vector for x in blue_tokens]
    blue_words = [x.text.lower() for x in blue_tokens]
    red_vectors = [x.vector for x in red_tokens]
    all_words_tokens = generate_all_words_tokens()
    
    ''' Find the clue word and blue word set that maximizes the score, i.e.
    lambda*helpfulness + (1-lambda)*unharmfulness '''
    max_score = (-1)*math.inf
    best_clue_word = None
    chosen_blue_word_tokens = None
    
    # Generate the SVM model
    if use_svm_unharmfulness:
        my_print('Generating svm model...')
        svm_model = generate_svm_model(blue_vectors,red_vectors)
        
    # Go over all the clue words, and choose the best one
    for cur_clue_word_token in all_words_tokens:
        if cur_clue_word_token.text.lower() in blue_words:
            continue # It is illegal to give a word which is on the board
        
        # Calculate helpfulness
        cur_helpfulness,best_word_token_set = get_best_blue_word_set(helpfulness_func, cur_clue_word_token, blue_tokens)
        
        # Calculate unharmfulness
        if use_svm_unharmfulness:
            cur_unharmfulness = svm_based_unharmfulness(svm_model, np.reshape(cur_clue_word_token.vector,(cur_clue_word_token.vector.shape[0],1)))
        else:
            cur_unharmfulness = distance_unharmfulness(red_vectors, cur_clue_word_token.vector)
            
        cur_score = LAMBDA * cur_helpfulness + (1-LAMBDA)*cur_unharmfulness
        if cur_score > max_score:
            max_score = cur_score
            best_clue_word = cur_clue_word_token
            chosen_blue_word_tokens = best_word_token_set
            
    print('Clue word: ' + str(best_clue_word) + ', referred blue words: ' + str(chosen_blue_word_tokens))
    print('All the blue words: ' + str(blue_tokens))
    print('All the red words: ' + str(red_tokens))
    
generate_clue(6, helpfulness1, True)