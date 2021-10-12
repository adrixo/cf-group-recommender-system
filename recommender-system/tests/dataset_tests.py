from recommender.dataset import Dataset
from configuration import *

import datetime
start_time = 0
total_time = 0

print("Dataset Module testing script.")

dataset_file = "../" + cfg['example_dataset']
columns = cfg['columns']
del_columns = cfg['del_columns']
sep = cfg['sep']

test_n = 1
print(test_n, ") Carga sin 3 columnas - error si no está preprocesado")
test_n += 1
start_time = datetime.datetime.now()
try:
    dm = Dataset(dataset_file, sep=sep)
except Exception as e:
    print(e)
    print(dir(e))
total_time = datetime.datetime.now() - start_time
print("Tiempo:", total_time)

print(test_n, ") Carga con 3 columnas")
test_n += 1
start_time = datetime.datetime.now()
try:
    dm = Dataset(dataset_file, sep=sep, cols=columns)
except Exception as ex:
    print(ex)
total_time = datetime.datetime.now() - start_time
print("Tiempo:", total_time)

print(test_n, ") Carga con eliminación columnas restantes")
test_n += 1
start_time = datetime.datetime.now()
try:
    dm = Dataset(dataset_file, sep=sep, del_cols=del_columns)
except Exception as ex:
    print(ex)
total_time = datetime.datetime.now() - start_time
print("Tiempo:", total_time)

print(test_n, ") Carga con eliminación y splitting")
test_n += 1
start_time = datetime.datetime.now()
try:
    dm = Dataset(dataset_file, sep=sep, del_cols=del_columns, split_dataset=True)
except Exception as ex:
    print(ex)
total_time = datetime.datetime.now() - start_time
print("Tiempo:", total_time)

