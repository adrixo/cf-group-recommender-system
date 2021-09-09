import sys
sys.path.append("../")

from recommender.dataset import Dataset
from recommender.algorithm import Algorithm
from recommender.group import Group
from recommender.aggregator import Aggregator

from configuration import *
random = [{'name':'random_0','members':['78267767','75886822','8297771667','951987','93551527']},{'name':'random_1','members':['64971753','91098385','31645756','96591785','785978']},{'name':'random_2','members':['3282418656','6261369445','5872573314','6745355488','4751371']},{'name':'random_3','members':['797223','6927814','8896336335','81244522','35936356']},{'name':'random_4','members':['14979344','59161244','8523819752','4697156726','52108718']},{'name':'random_5','members':['951522','470146','3668878158','4627113','933257765']},{'name':'random_6','members':['82713369','951987','127168','5256266264','1942884394']},{'name':'random_7','members':['34186093','4776242312','26916976','18878516','933219667']},{'name':'random_8','members':['2548126169','45491911','56476038','5899281147','3878928761']},{'name':'random_9','members':['9141421346','5627565258','67325998','4578536541','5175473351']}]
chill = [{'name':'chill_0','members':['1149285','1869532814','54048864','904896','1796685823']},{'name':'chill_1','members':['71614496','1779773528','36835636','4931748929']},{'name':'chill_2','members':['54048864','4596621995','855166','82144649','941355']},{'name':'chill_3','members':['52998996','45797321','82842732','4713724134','51858248']},{'name':'chill_4','members':['815642','19661454','8896336335','2124836141','545376']},{'name':'chill_5','members':['84922213','6180','56855','3287287687','4328165168']},{'name':'chill_6','members':['23386159','84922213','7361','23438939']},{'name':'chill_7','members':['31341725','62354634','8154277998','4285411697','941355']},{'name':'chill_8','members':['9151943674','31645756','88255851','6358665743','6783438985']},{'name':'chill_9','members':['4559399978','66499973','904896','14859564','26865185']}]
fitness = [{'name':'fitness_0','members':['825734','68371256','27257479','27995534','9977563423']},{'name':'fitness_1','members':['42824321','82144649','7316678','96522825','3387352']},{'name':'fitness_2','members':['53923822','1380541','3236412584','71342467','27995534']},{'name':'fitness_3','members':['16594348','87891768','332265632','84893643']},{'name':'fitness_4','members':['1898698556','6864587558','4559399978','1318281982','51621974']},{'name':'fitness_5','members':['456932','1956544','5687852687','9312188','282780']},{'name':'fitness_6','members':['1869532814','8329586','9453368746','1318281982','77566421']},{'name':'fitness_7','members':['5735792','6124439292','1956544','4624348449','6332857992']},{'name':'fitness_8','members':['1185137848','319152','9511146','3296375665','34186093']},{'name':'fitness_9','members':['257956368','794756','36835636','6864587558','811514633']}]
party = [{'name':'party_0','members':['58229675','8154277998','5675148268','1149285','7449921858']},{'name':'party_1','members':['245556','24725485','9151943674','59123','36835636']},{'name':'party_2','members':['191832714','33936926','815642','9157941164','3879214236']},{'name':'party_3','members':['15571839','779235','54048864','855166','56855']},{'name':'party_4','members':['13475539','5687852687','21311472','9173652611','5923588368']},{'name':'party_5','members':['4559399978','51858248','3282418656','6974815126','88255851']},{'name':'party_6','members':['7449921858','3128861823','4514137645','3296375665','68771694']},{'name':'party_7','members':['855166','123163','63974474','1166562392','84922213']},{'name':'party_8','members':['3139118561','4285411697','521549','2494216246','82144649']},{'name':'party_9','members':['4713724134','73983959','8392953896','62522328','88699738']}]


dataset_file = cfg['example_dataset']
sep = cfg['sep']
columns = cfg['columns']
rating_scale = cfg['rating_scale']
relevant_threshold = cfg['relevant_threshold']
lib = cfg['lib']
verbose = cfg['verbose']
algorithm = cfg['algorithm']
algorithms = cfg['surprise_algorithm_col']
aggregation_methods = cfg['aggregation_methods']
k_tests = cfg['experiment_k_depth']

contexts_colection_def = [ cfg['no_context'], cfg['chill_context'], cfg['fitness_context'],cfg['party_context']]

custom_group = cfg['example_group']
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



