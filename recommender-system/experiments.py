from recommender.dataset import Dataset
from recommender.algorithm import Algorithm
from recommender.group import Group
from recommender.aggregator import Aggregator

from configuration import *

dataset_file = cfg['experiment_dataset']
sep = cfg['sep']
columns = cfg['columns']
rating_scale = cfg['rating_scale']
relevant_threshold = cfg['relevant_threshold']
lib = cfg['lib']
verbose = cfg['verbose']
algorithms = cfg['surprise_algorithm_col']
aggregation_methods = cfg['aggregation_methods']
k_tests = cfg['experiment_k_depth']

contexts_colection_def = [ cfg['no_context'], cfg['chill_context'], cfg['fitness_context'],cfg['party_context']]

groups_colection = cfg['experiment_groups']


test_n = 0
successful_tests = 0
start_tests_time()

#ALTERNATIVA Para distintos thresholds de considerar item positivo en la evaluación
#://urpsrise.readthedocs.io/en/stable/getting_started.html#use-cross-validation-iterators
#kf = KFold(random_state=0)  # folds will be the same for all algorithms.


#10 grupos aleatorios y relacionados

# Inicia el archivo de guardado
results_file = "results/results_file.csv"
desired_metrics = ["precision", "recall",
#                    "mae", "rmse", "mae_2", "rmse_2",
                    "ndcg"]
                    #["precision_2", "recall_2", "serendipity", "coverage", "consensus", "fairness"]
results_file_columns = ["timestamp", "context_type", "group_type", "algorithm", "aggregation_method", "@k"] + desired_metrics
redo = save_experiment_results_header(results_file, results_file_columns, sep=sep)

# 1. Para cada contexto
try:
    for context in contexts_colection_def: 
        dm = Dataset(dataset_file, 
                        sep=sep, 
                        rating_scale=rating_scale, 
                        cols=columns, 
                        split_dataset=True, 
                        lib=lib, 
                        test_size=0.25,
                        line_format='user item rating', 
                        prefilter_columns=context["context"],
                        verbose=verbose)

        try:
            # 2. Para cada algoritmo
            for alg_it in algorithms:

                #PARA CADA FOLD DE CROSS VALIDATION:
                print("\n######## "+alg_it+" ########\n")
                am = Algorithm(dm, lib=lib, algorithm=alg_it)
                am.fit_model()

                # 3. Para cada grupo
                for group_it in groups_colection:
                    group_members = group_it["members"]
                    group_name = group_it["name"]
                    group = Group(group_name=group_name, group_context_name=context["name"])
                    group.add_list_of_users(group_members)
                    group.print_users()

                    # Si es un grupo contextual, solo ejecutar el experimento en su contexto y sin contexto
                    if context["name"] != 'none' and (group_it["name"].split('_')[0] != context["name"] or context["name"] == 'none'):
                        pass
                    else:
                        # 4. Para cada método de agregación
                        for agg in aggregation_methods:
                            try:
                                test_n = new_tests(test_n,  alg_it + "- Metodo agregación " + agg)
                                aggregation = Aggregator(dm, am, group, aggregation_method=agg)

                                print("Performing recomendation")
                                rec = aggregation.perform_group_recommendation()
                                print("Performed group recommendation.")
                                finis_test()


                                # 5. Para cada l to ongitud top-N
                                print("Calculando rango top-n: ", k_tests)
                                metrics = aggregation.evaluate(threshold=relevant_threshold, k=k_tests)

                                for k in range(k_tests[0],k_tests[1],k_tests[2]):

                                    c_metrics = []
                                    for metric_name in desired_metrics:
                                        metric_exists = False
                                        for metric in metrics:
                                            #Si es el k correcto
                                            if metric['k']==k:
                                                if metric['metric_name'] == metric_name:
                                                    metric_exists = True
                                                    c_metrics.append(metric['metric_value'])
                                        if not metric_exists:
                                            c_metrics.append("ukn") # No se ha generado esa metrica y se deseaba

                                                
                                    # 6. Almacenaje de los resultados
                                    result_to_save = [datetime.datetime.now(), context["name"], group_name, alg_it, agg, k] + c_metrics
                                    save_experiment_results(results_file, result_to_save, results_file_columns, sep=sep)
                                    successful_tests += 1 
                                print()
                            
                            except Exception as ex:
                                # Imprimir en el archivo en que momento salió el error
                                print("Error algoritmo " + alg_it + " - agregación: " + agg , ex)
                                current_variables = [datetime.datetime.now(), context["name"], group_name, alg_it, agg, "k"]
                                err = ["err" for f in range(len(results_file_columns)-len(current_variables))]
                                result_to_save = current_variables + err
                                save_experiment_results(results_file, result_to_save, results_file_columns, sep=sep)

        except Exception as ex:
            print("Error fitting with alg: " + alg_it, ex)
            current_variables = [datetime.datetime.now(), context["name"], group_name, alg_it]
            err = ["err" for f in range(len(results_file_columns)-len(current_variables))]
            result_to_save = current_variables + err
            save_experiment_results(results_file, result_to_save, results_file_columns, sep=sep)


except Exception as ex:
    print("Error with context ex: ", ex)
    current_variables = [datetime.datetime.now(), context["name"]]
    err = ["err" for f in range(len(results_file_columns)-len(current_variables))]
    result_to_save = current_variables + err
    save_experiment_results(results_file, result_to_save, results_file_columns, sep=sep)




tests_total_time()
successfull_tests_print(successful_tests, test_n)



