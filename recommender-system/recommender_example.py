from recommender.dataset import Dataset
from recommender.algorithm import Algorithm
from recommender.group import Group
from recommender.aggregator import Aggregator

from configuration import *

# Configuration setup
dataset_file = cfg['example_dataset']
sep = cfg['sep']
rating_scale = cfg['rating_scale']
columns = cfg['columns']
split_dataset = cfg['split_dataset']
lib = cfg['lib']
line_format = cfg['line_format']
verbose = cfg['verbose']
# Algorithm
algorithm = cfg['algorithm']
# Group
group_name = cfg['example_group_name']
group_context_name = cfg['example_group_context']
custom_group_members = cfg['example_group']

# 1. Dataset load
print("Data loading into dataset")
ds = Dataset(
    dataset_file, sep=sep, cols=columns,
    rating_scale=rating_scale, split_dataset=True,
    lib=lib, line_format=line_format, 
    verbose=verbose)

# 1.optional: Contextual prefiltering


# 2. Model load and fitting
alg = Algorithm(ds, lib, algorithm)
alg.fit_model()

# 3. Group creation
group = Group(group_name=group_name, group_context_name=group_context_name)
group.add_list_of_users(custom_group_members)

# 4. Aggregation of group recommendation prediction
agg = Aggregator(ds, alg, group)
rec = agg.perform_group_recommendation()
agg.print_group_recommendation()

# 5. Group recommendation evaluation
agg.evaluate()
