from recommender.dataset import Dataset
from recommender.algorithm import Algorithm
from recommender.group import Group
from recommender.aggregator import Aggregator

from configuration import *

dataset_file = "../" + cfg['example_dataset']
sep = cfg['sep']
columns = cfg['columns']
rating_scale = cfg['rating_scale']
lib = cfg['lib']
verbose = cfg['verbose']
algorithm = cfg['algorithm']
custom_group = cfg['example_group']
aggregation_methods = cfg['aggregation_methods']

successful_tests = 1
test_n = 1
start_tests_time()

dm = Dataset(dataset_file, sep=sep, rating_scale=rating_scale, cols=columns, split_dataset=True, lib=lib, line_format='user item rating', verbose=verbose)
am = Algorithm(dm, lib=lib, algorithm=algorithm)
am.fit_model()

group = Group(group_name="Test1", group_context_name="Random")
group.add_list_of_users(custom_group)

# Prueba de todos los distintos métodos de agregación con surprise:
for agg in aggregation_methods:
    test_n = new_tests(test_n, "Testing aggregation method " + agg)
    try:
        print("\n\tAggregation_method: "+agg)
        aggregation = Aggregator(dm, am, group, aggregation_method=agg)
        rec = aggregation.perform_group_recommendation()
        print(list(rec.items())[:100])
        successful_tests += 1 
    except Exception as ex:
        print(ex)
finis_test()

tests_total_time()
successfull_tests_print(successful_tests, test_n)
