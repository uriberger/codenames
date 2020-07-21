from unharmfulness import distance_unharmfulness,svm_based_unharmfulness,generate_svm_model
import pandas as pd
import numpy as np
import spacy
import math

debug = True
def my_print(str):
    if debug:
        print(str)

board_filename = 'game_words.xlsx'
all_words_filename = 'all_words.txt'
my_print('Loading nlp model...')
nlp = spacy.load("en_core_web_md")

def generate_game_word_sets(game_number):
    df = pd.read_excel(board_filename)
    column_list = df.columns.ravel()
    all_words = df['words'].tolist()
    blue_words = [all_words[i] for i in range(len(all_words)) if df['colors'][i] == 'blue' and df['game number'][i] == game_number]
    blue_words = ['apple','banana','apricot','watermelon','grapefruit']
    red_words = [all_words[i] for i in range(len(all_words)) if df['colors'][i] == 'red' and df['game number'][i] == game_number]
    red_words = ['tiger','dog','bird','elephant','mouse']
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
        
blue_tokens,red_tokens = generate_game_tokens(1)
blue_vectors = [x.vector for x in blue_tokens]
blue_words = [x.text for x in blue_tokens]
red_vectors = [x.vector for x in red_tokens]
all_words_tokens = generate_all_words_tokens()
max_unharmfulness = (-1)*math.inf
best_clue_word = None

my_print('Generating svm model...')
svm_model = generate_svm_model(blue_vectors,red_vectors)
i = 0
for cur_clue_word_token in all_words_tokens:
    if cur_clue_word_token.text in blue_words:
        continue
    #cur_unharmfulness = distance_unharmfulness(red_vectors, cur_clue_word_token.vector)
    cur_unharmfulness = svm_based_unharmfulness(svm_model, np.reshape(cur_clue_word_token.vector,(300,1)))
    if cur_unharmfulness > max_unharmfulness:
        max_unharmfulness = cur_unharmfulness
        best_clue_word = cur_clue_word_token.text
    i += 1
print(best_clue_word)