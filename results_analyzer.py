#################################
####### Codenames project #######
# By Uri Berger, Tomer Genossar #
########## August 2020 ##########
#################################

'''
File name: results_analyzer.py.
Description: This file collects and analyzes the results of the experiment.
'''

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# File names

board_filename = 'game_words.xlsx'
response_prefix = 'Codenames clues '
response_suffix = '.csv'
clues_log_prefix = 'clues'
clues_log_suffix = '.csv'

# Parameters

MODEL_NUM = 4
GAMES_NUM = 25
FORM_NUM = 5

def generate_game_word_sets(game_number):
    ''' Generate the blue and red word lists, for the given game number. '''
    df = pd.read_excel(board_filename)
    column_list = df.columns.ravel()
    all_words = df['words'].tolist()
    blue_words = [all_words[i].lower().replace(' ','_') for i in range(len(all_words)) if df['colors'][i] == 'blue' and df['game number'][i] == game_number]
    red_words = [all_words[i].lower().replace(' ','_') for i in range(len(all_words)) if df['colors'][i] == 'red' and df['game number'][i] == game_number]
    return blue_words,red_words

def generate_referred_blue_words(game_number,model_num):
    ''' Generate the list of blue words that the clue, generates by model number 'model_num',
    refers to. This function only works after the function 'generate_clue' from the
    game_simulation.py file was executed, with the WRITE_CLUES_TO_FILE flag on. '''  
    clues_filename = clues_log_prefix + str(model_num) + clues_log_suffix
    with open(clues_filename, newline='') as csvfile:
        clues_reader = csv.reader(csvfile)
        first = True
        for row in clues_reader:
            if first:
                first = False
                continue
            if int(row[0]) == game_number:
                return row[2].split(';')
    print('Game not found!')
    assert False

def get_possible_grades():
    ''' Generate the list of possible grades, for the sake of printing CDFs.
    We choose some small delta, and take all the possible grades in the possible grades range
    [-1,1.5] with delta differences. '''
    delta = 0.1
    possible_grades_range_begin = -1
    possible_grades_range_len = 2.5
    possible_grades = [possible_grades_range_begin + delta*x for x in range(int(possible_grades_range_len/delta)+1)]
    return possible_grades
    
def cdf(my_list):
    ''' Generate the CDF of the given list. '''
    res_cdf = []
    possible_grades = get_possible_grades()
    for grade in possible_grades:
        num_up_to_cur_grade = len([y for y in my_list if y <= grade])
        res_cdf.append(num_up_to_cur_grade)

    return res_cdf

def collect_results_for_game(form_number, form_game_number,orig_game_number,model_num):
    ''' Collect the grades for a given game.
    Form number is the number of the Google Form in which the game is located, form_game_number is
    the location of the game within the form, orig_game_number is the number of the game in the
    original games file, and model_num is the number of the model used to generate the clue for
    this game.
    This function can only work after the Google Forms responses were exported to csv and copied
    to the script's directory. '''
    blue_words,red_words = generate_game_word_sets(orig_game_number)
    referred_blue_words = generate_referred_blue_words(orig_game_number, model_num)
    referred_blue_words_num = len(referred_blue_words)
    
    # Statitics
    empty_answers_num = 0
    incorrect_num_of_words_num = 0
    grades = []
    
    response_filename = response_prefix + str(form_number) + str(response_suffix)
    with open(response_filename, newline='') as csvfile:
        response_reader = csv.reader(csvfile)
        first = True
        for row in response_reader:
            if first:
                first = False
                continue
            
            answers = row[form_game_number].split(';')
            if len(answers) == 1 and len(answers[0]) == 0:
                empty_answers_num += 1
                continue
            if len(answers) != referred_blue_words_num:
                incorrect_num_of_words_num += 1
                continue
            
            answers = [x.lower() for x in answers]
            grade = 0
            for answer in answers:
                if answer in referred_blue_words:
                    grade += 1.5
                elif answer in blue_words:
                    grade += 1
                elif answer in red_words:
                    grade -= 1
            grades.append(grade/referred_blue_words_num) # Grades are normalized according to the number of referred blue words

    return grades,empty_answers_num,incorrect_num_of_words_num
         
def collect_statistics():
    ''' Collect different statistics for the experiment. '''   
    legal_answers_per_game = {x:0 for x in range(1,GAMES_NUM+1)}
    legal_answers_per_model = {x:0 for x in range(1,MODEL_NUM+1)}
    grades_per_game = {x:[] for x in range(1,GAMES_NUM+1)}
    grades_per_model = {x:[] for x in range(1,MODEL_NUM+1)}
    illegal_answers_per_game = {x:0 for x in range(1,GAMES_NUM+1)}
    illegal_answers_per_model = {x:0 for x in range(1,MODEL_NUM+1)}
    empty_answers_per_game = {x:0 for x in range(1,GAMES_NUM+1)}
    empty_answers_per_model = {x:0 for x in range(1,MODEL_NUM+1)}
    
    ''' The forms are built in the following manner:
    Form 1:
    - Games 1-5 with model 1
    - Games 6-10 with model 2
    - Games 11-15 with model 3
    - Games 16-20 with model 4
    Form 2:
    - Games 6-10 with model 1
    - Games 11-15 with model 2
    - Games 16-20 with model 3
    - Games 21-25 with model 4
    Form 3:
    - Games 1-5 with model 4
    - Games 11-15 with model 1
    - Games 16-20 with model 2
    - Games 21-25 with model 3
    
    etc. Each form has the form games numbers (for example, in form 3rd game is game number 3
    with model 1 and the 6th game is game number 11 with model 1), and each of this game has its
    original game number. '''
    for form_number in range(1,FORM_NUM+1):
        ''' First, generate the "all_model_games" data base: For each model, generate a list of
        tuples (form_game_number, orig_game_number), of all the games it has in the current form. '''
        if form_number == 1:
            model_1_form_games = range(1,6)
        else:
            model_1_form_games = range(1 + (form_number-2)*5,1 + (form_number-1)*5)
        model_1_orig_games = range(1 + (form_number-1)*5,1 + form_number*5)
        model_1_games = [(model_1_form_games[i],model_1_orig_games[i]) for i in range(len(model_1_orig_games))]
        
        all_model_games = [model_1_games]
        for _ in range(MODEL_NUM-1):
            all_model_games.append([(((x[0]+4) % 20)+1,((x[1]+4) % 25)+1) for x in all_model_games[-1]])
        
        for model_number in range(1,MODEL_NUM+1):
            for form_game_number,orig_game_number in all_model_games[model_number-1]:
                grades,empty_answers_num,incorrect_num_of_words_num = collect_results_for_game(form_number, form_game_number, orig_game_number, model_number)
                
                legal_answers_per_game[orig_game_number] += len(grades)
                legal_answers_per_model[model_number] += len(grades)
                grades_per_game[orig_game_number] += grades
                grades_per_model[model_number] += grades
                illegal_answers_per_game[orig_game_number] += incorrect_num_of_words_num
                illegal_answers_per_model[model_number] += incorrect_num_of_words_num
                empty_answers_per_game[orig_game_number] += empty_answers_num
                empty_answers_per_model[model_number] += empty_answers_num
                
    return \
        legal_answers_per_game, legal_answers_per_model, \
        grades_per_game, grades_per_model, \
        illegal_answers_per_game, illegal_answers_per_model, \
        empty_answers_per_game, empty_answers_per_model
        
legal_answers_per_game, legal_answers_per_model, \
grades_per_game, grades_per_model, \
illegal_answers_per_game, illegal_answers_per_model, \
empty_answers_per_game, empty_answers_per_model = collect_statistics()

# Generate different plots

# Generate a plot presenting the number of answers per original game number
legal_answers_range = range(1,GAMES_NUM+1)
illegal_answers_range = [x+0.25 for x in legal_answers_range]
empty_answers_range = [x+0.5 for x in legal_answers_range]
plt.bar(legal_answers_range, legal_answers_per_game.values(), color = 'b', width = 0.25)
plt.bar(illegal_answers_range, illegal_answers_per_game.values(), color = 'r', width = 0.25)
plt.bar(empty_answers_range, empty_answers_per_game.values(), color = 'k', width = 0.25)
plt.legend(labels=['Legal answers', 'Illegal answers', 'Empty answers'])
plt.title('Number of answers per game')
plt.xlabel('Game number')
plt.ylabel('Number of answers')
plt.xticks(range(1,GAMES_NUM+1))
plt.show()

# Generate a plot presenting the number of answers per model
plt.clf()
legal_answers_range = range(1,MODEL_NUM+1)
illegal_answers_range = [x+0.25 for x in legal_answers_range]
empty_answers_range = [x+0.5 for x in legal_answers_range]
plt.bar(legal_answers_range, legal_answers_per_model.values(), color = 'b', width = 0.25)
plt.bar(illegal_answers_range, illegal_answers_per_model.values(), color = 'r', width = 0.25)
plt.bar(empty_answers_range, empty_answers_per_model.values(), color = 'k', width = 0.25)
plt.legend(labels=['Legal answers', 'Illegal answers', 'Empty answers'])
plt.title('Number of answers per model')
plt.xlabel('Model number')
plt.ylabel('Number of answers')
plt.xticks(range(1,MODEL_NUM+1))
plt.show()

# Generate a plot presenting the mean grade per original game number
plt.clf()
game_range = range(1,GAMES_NUM+1)
game_average_data = [sum(grades_per_game[x])/legal_answers_per_game[x] for x in range(1,GAMES_NUM+1)]
game_std_data = [np.std(grades_per_game[x]) for x in range(1,GAMES_NUM+1)]
plt.bar(game_range, game_average_data, yerr = game_std_data, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.title('Mean grade per game')
plt.xlabel('Game number')
plt.ylabel('Mean grade')
plt.xticks(range(1,GAMES_NUM+1))
plt.show()

# Generate a plot presenting the mean grade per model
plt.clf()
model_range = range(1,MODEL_NUM+1)
model_average_data = [sum(grades_per_model[x])/legal_answers_per_model[x] for x in range(1,MODEL_NUM+1)]
model_std_data = [np.std(grades_per_model[x]) for x in range(1,MODEL_NUM+1)]
plt.bar(model_range, model_average_data, yerr = model_std_data, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.title('Mean grade per model')
plt.xlabel('Model number')
plt.ylabel('Mean grade')
plt.xticks(range(1,MODEL_NUM+1))
plt.show()

# Generate a plot presenting the CDF of grades per model
plt.clf()
cdfs = {x:cdf(grades_per_model[x]) for x in range(1,MODEL_NUM+1)}
for model_ind in range(1,MODEL_NUM+1):
    plt.plot(get_possible_grades(), cdfs[model_ind])
plt.legend(['Model ' + str(x) for x in range(1,MODEL_NUM+1)])
plt.title('Grade CDF per model')
plt.xlabel('Grade')
plt.ylabel('Number of answers up to this grade')
plt.show()

# Conduct statistic tests

# T-test
svm_models_grades = grades_per_model[1] + grades_per_model[2]
non_svm_models_grades = grades_per_model[3] + grades_per_model[4]
svm_test_res = stats.ttest_ind(svm_models_grades,non_svm_models_grades)
print('According to t-test:')
if svm_test_res[0] > 0:
    print('\tSVM models are better, with p-value ' + str(svm_test_res[1]))
else:
    print('\tnon-SVM models are better, with p-value ' + str(svm_test_res[1]))
    
# Mann-Whitney test
_,p_value_svm_is_better = stats.mannwhitneyu(svm_models_grades, non_svm_models_grades, alternative='greater')
_,p_value_non_svm_is_better = stats.mannwhitneyu(svm_models_grades, non_svm_models_grades, alternative='less')
print('According to Mann-Whitney test:')
if p_value_svm_is_better < p_value_non_svm_is_better:
    print('\tSVM models are better, with p-value ' + str(p_value_svm_is_better))
else:
    print('\tnon-SVM models are better, with p-value ' + str(p_value_non_svm_is_better))