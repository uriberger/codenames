import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

board_filename = 'game_words.xlsx'
response_prefix = 'Codenames clues '
response_suffix = '.csv'
clues_log_prefix = 'clues'
clues_log_suffix = '.csv'

MODEL_NUM = 4
GAMES_NUM = 25
FORM_NUM = 5

def generate_game_word_sets(game_number):
    df = pd.read_excel(board_filename)
    column_list = df.columns.ravel()
    all_words = df['words'].tolist()
    blue_words = [all_words[i].lower().replace(' ','_') for i in range(len(all_words)) if df['colors'][i] == 'blue' and df['game number'][i] == game_number]
    red_words = [all_words[i].lower().replace(' ','_') for i in range(len(all_words)) if df['colors'][i] == 'red' and df['game number'][i] == game_number]
    return blue_words,red_words

def generate_referred_blue_words(game_number,model_num):
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

def collect_results_for_game(form_number, form_game_number,orig_game_number,model_num):
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
            grades.append(grade/referred_blue_words_num)
    return grades,empty_answers_num,incorrect_num_of_words_num
            
legal_answers_per_game = {x:0 for x in range(1,GAMES_NUM+1)}
legal_answers_per_model = {x:0 for x in range(1,MODEL_NUM+1)}
grades_per_game = {x:[] for x in range(1,GAMES_NUM+1)}
grades_per_model = {x:[] for x in range(1,MODEL_NUM+1)}
illegal_answers_per_game = {x:0 for x in range(1,GAMES_NUM+1)}
illegal_answers_per_model = {x:0 for x in range(1,MODEL_NUM+1)}
empty_answers_per_game = {x:0 for x in range(1,GAMES_NUM+1)}
empty_answers_per_model = {x:0 for x in range(1,MODEL_NUM+1)}

for form_number in range(1,FORM_NUM+1):
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