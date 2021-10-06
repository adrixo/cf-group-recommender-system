import datetime
import sys
sys.path.append(".")
dataset_path = 'datasets/'

from datasets.experiment_groups import *

cfg = {
    'verbose'       : True,
# Datasets:
    'example_dataset' : dataset_path + 'example_dataset.csv',
    'experiment_dataset' : dataset_path + 'reduced_50.csv',
    'large_dataset' : dataset_path + 'test_large.csv',
    'custom_dataset' : dataset_path + 'custom.csv',
# Dataset configuration:
    'sep'           : '\t',
    'columns'       : ["uid", "iid", "rating"],
    'split_dataset' : True,
    'del_columns'   : ["user", "spotify_track_id", "track_name", "artist_name", "play_count_track", "play_count_total_user", "explicit", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms"],
    'line_format'   : 'user item rating',
    'rating_scale'  : (0,4),
    'kfold_n': 1,
    'relevant_threshold'  : 0.6,
    'experiment_k_depth' : (5,101,5),
# groups
    'example_group' : ["2", "3", "4"],
    'example_group_name' : 'example_group',
    'example_group_context' : 'example_context',
    'experiment_groups': random + chill + fitness + party,
    'custom_user'   : "2",
# Librery and algorithms
    'lib'           : 'surprise',
    'libs'          : ['surprise'],
    'algorithm'     : 'svd',
    'surprise_algorithm_col' : ["coclustering", "svd"],
    'aggregation_methods': ["avg","add","app","avm","lms","maj","mpl","mul"],
    'implemented_aggregation_methods': ["avg","add","app","avm","lms","maj","mpl","mul"],
# Contexts
    'custom_contexts': [{"column": 'acousticness', "mode": "value", "threshold": 0.2, "direction": "below"}],
    'no_context': {"name":"none",
                      "context": []
                    },
    'party_context': {"name":"party",
                      "context": [{"column": 'valence', "mode": "value", "threshold": 0.85, "direction": "above"}]
                    },
    'chill_context': {"name":"chill",
                      "context": [{"column": 'energy', "mode": "value", "threshold": 0.3, "direction": "below"}]
                    },
    'fitness_context': {"name":"fitness",
                      "context": [{"column": 'energy', "mode": "value", "threshold": 0.94, "direction": "above"}
                      ]
                    },
}

start_time = 0
total_time = 0
ind_test_start_time = 0

def new_tests(test_n, tests_text):
    global ind_test_start_time
    ind_test_start_time = datetime.datetime.now()
    print("\n" + str(test_n+1) + ") " + tests_text)
    return test_n + 1

def finis_test(test_n="", tests_text=""):
    global ind_test_start_time
    test_time = datetime.datetime.now() - ind_test_start_time
    print("Test finished in : " + str(test_time))

def successfull_tests_print(succesfull_test, test_n):
    global total_time
    print(str(succesfull_test) + "/" + str(test_n) + " test passed in " + str(total_time) + "s" )

def start_tests_time():
    global start_time
    start_time = datetime.datetime.now()

def tests_total_time():
    global total_time
    total_time = datetime.datetime.now() - start_time

def save_experiment_results_header(file, columns, sep="\t"):  
    f = open(file, "w")

    for col in columns:
        f.write(str(col) + sep)

    f.write("\n")
    f.close()

    return True

def save_experiment_results(file, columns, header_columns, sep="\t"):
    
    if len(columns) != len(header_columns):
        print("La longitud de las columnas no coincide", len(columns), len(header_columns))
        print(columns)
        print(header_columns)
        input()
        return

    f = open(file, "a")

    for col in columns:
        f.write(str(col) + sep)
    f.write("\n")
    f.close()
