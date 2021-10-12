from recommender.dataset import Dataset
from recommender.algorithm import Algorithm

from configuration import *

print("Dataset Module testing script.")

dataset_file = "../" + cfg['example_dataset']
columns = cfg['columns']
del_columns = cfg['del_columns']
sep = cfg['sep']
lib = cfg['lib']
algorithm = cfg['algorithm']
algorithms = cfg['surprise_algorithm_col'] # TODO: colección de algoritmos para cada librería, o consistencia
user_id_to_test = cfg['custom_user']
context = cfg['no_context']
rating_scale = cfg['rating_scale']

successful_tests = 0
test_n = 0

start_tests_time()

#CARGA dataset
test_n = new_tests(test_n, "Se carga dataset con eliminación y splitting.")
try:
    dm = Dataset(dataset_file, 
                        sep=sep, 
                        rating_scale=rating_scale, 
                        cols=columns,
                        split_dataset=True, 
                        test_size=0.25,
                        prefilter_columns=context,
                        lib=lib, 
                        line_format='user item rating')
    successful_tests += 1
except Exception as ex:
    print(ex)
finis_test()

# El dataset ya estaría fijo y tan solo se modificaría el modelo del algorithm (comprobar)
## dataset_module_is_correct
test_n = new_tests(test_n, "Carga dataset terminada \n Comprobación carga algoritmo simple.")
try:
    am = Algorithm(dm, lib=lib, algorithm=algorithm)
    successful_tests += 1
except Exception as ex:
    print(ex)
finis_test()

# Como el anterior pero con entrenamiento
test_n = new_tests(test_n, "Comprobación entrenamiento algoritmo simple.")
try:
    am = Algorithm(dm, lib=lib, algorithm=algorithm)
    am.fit_model()
    successful_tests += 1
except Exception as ex:
    print(ex)
finis_test()

## fit_model + top-n
test_n = new_tests(test_n, "Comprobación algoritmo simple + Top-N unica persona.")
try:
    #Metodo mas simple a la hora de gestionar la lista
    am = Algorithm(dm, lib=lib, algorithm=algorithm)
    am.fit_model()
    top_n_list = am.get_top_n_list_simple_user(user_id_to_test)
    print(top_n_list)
    successful_tests += 1
except Exception as ex:
    print(ex)
finis_test()

## fit_model + top-n
test_n = new_tests(test_n, "Comprobación algoritmo simple + Top-N unica persona de forma externa")
try:
    # Método mas rapido para tener las listas de todos los usuarios
    am = Algorithm(dm, lib=lib, algorithm=algorithm)
    am.fit_model()
    top_n_list = am.get_top_n_list(user_id_to_test)
    am.print_top_n_list(top_n_list, user_id_to_test)
    successful_tests += 1
except Exception as ex:
    print(ex)
finis_test()

## do_cross_validate_model
test_n = new_tests(test_n, "Comprobación algoritmo simple + crossvalidate.")
try:
    am = Algorithm(dm, lib=lib, algorithm=algorithm)
    am.do_cross_validate_model()
    successful_tests += 1
except Exception as ex:
    print(ex)
finis_test()

## Check all the surprise algorithms fit
test_n = new_tests(test_n, "Comprobación de todos los algoritmos de " + lib)
try:
    for alg_it in algorithms:
        am = Algorithm(dm, lib=lib, algorithm=alg_it)
        am.fit_model()
    successful_tests += 1
except Exception as ex:
    print(ex)
finis_test()
    
## Check all the surprise algorithms crossvalidate
test_n = new_tests(test_n, "Comprobación de todos los algoritmos de " + lib)
try:
    for alg_it in algorithms:
        am = Algorithm(dm, lib=lib, algorithm=alg_it)
        am.do_cross_validate_model()
    successful_tests += 1
except Exception as ex:
    print(ex)
finis_test()

tests_total_time()
successfull_tests_print(successful_tests, test_n)
    