import pickle

from configuration import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('bmh')


dataset_file = '../../' + cfg['experiment_dataset']
sep = cfg['sep']
contexts_colection_def = [ cfg['fitness_context'], cfg['chill_context'],cfg['party_context']]
cols = cfg['columns']

n_groups = 3
group_size = 5

df = pd.read_csv(dataset_file, sep=sep)
print(df)
print(df.shape)
numero_valoraciones_total = df.shape[0]
users = df["uid"].unique().tolist()
items = df["iid"].unique().tolist()
n_users = len(users)-1
n_items = len(items)-1
print(len(users))

print("# users: ", n_users)
print("# items: ", n_items)
print("# ratings per user: ", numero_valoraciones_total/len(users))
print("# ratings per item: ", numero_valoraciones_total/len(items))
print("Items per user: ", len(items)/len(users))
print('\n\tUID:')
print(df['uid'].describe())
print('\n\tIID:')
print(df['iid'].describe())
print('\n\tRating:')
print(df['rating'].describe())
print()
# plt.figure(figsize=(9, 8))
# sns.distplot(df['rating'], color='g', bins=8, hist_kws={'alpha': 0.4})
# plt.show()
# plt.figure(figsize=(9, 8))
# sns.distplot(df['uid'], color='g', norm_hist=True, bins=20, hist_kws={'alpha': 0.4})
# plt.show()
# plt.figure(figsize=(9, 8))
# sns.distplot(df['iid'], color='g', bins=20, hist_kws={'alpha': 0.4})
# plt.show()

# Limpiamos para obtener solo los contextuales
dfcontext = df.copy()
del_cols = cols + ['play_count_track', 'play_count_total_user']
for del_column in del_cols:
    del dfcontext[del_column]
df_num = dfcontext.select_dtypes(include = ['float64', 'int64'])
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()

# # Limpiamos para que se haga análisis antes:
del_cols = list(df.columns)
for col in cols: 
    if col in del_cols : del_cols.remove(col)

for del_column in del_cols:
    del df[del_column]

df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()

# Items con muchos ratings
items_dict = {}
i = 0
print("Start analyzing items")
# Tomar los items y sus ratings
# for item in items:
#     # print("item " + str(i)+"/"+str(n_items))
#     items_dict[item] = {}
#     df_aux = df.loc[df['iid']==item].index
#     items_dict[item]["rated_times"] = df_aux.shape[0]
#     i+=1
# print("Done")
# a_file = open('eda/items_dict.pkl', "wb")
# pickle.dump(items_dict, a_file)
# a_file.close()
a_file = open("items_dict.pkl", "rb")
items_dict = pickle.load(a_file)
full_items_index = None
threshold = 50

items_density = {}
# print(items_dict.items())
for item in items_dict:

        try:
            items_density[items_dict[item]['rated_times']] += 1
        except Exception as e:
            items_density[items_dict[item]['rated_times']] = 1
items_density = {k: items_density[k] for k in sorted(items_density)}

n_rated_songs = []
n_times = []
for i in items_density:
    n_rated_songs.append(i)
    n_times.append(items_density[i])
tdf = pd.DataFrame({'n_ratings':n_rated_songs, 'ocurrences': n_times})  
print(tdf)
plt.figure(figsize=(9, 8))
ax = sns.lineplot(data=tdf, x="n_ratings", y="ocurrences", color='g')
ax.set(xlabel='Número de valoraciones por canción', ylabel='ocurrencias')
plt.show()

# Tomar las posiciones de aquellos usuarios con pocos ratings para evitar que tarde mucho
# total = len(items_dict)
# count = 0

# x = None

# for item in items_dict:
#     if items_dict[item]["rated_times"] > threshold:
#         count += 1
#         items_dict[item]["index"] = df.loc[df['iid']==item].index
#         try:
#             full_items_index = full_items_index.append(items_dict[item]["index"]) 
#         except Exception as e:
#             full_items_index = items_dict[item]["index"]
# print(count/total, count, total)
# print("index a mantener: ", full_items_index.shape)
# reversed_index = df.index.difference(full_items_index)
# df.drop(reversed_index, inplace=True)
# print(df.shape)
# df.to_csv('datasets/reduced_'+str(threshold)+'.csv', index=False, header=True, sep='\t')

#Eliminamos arhcivos menores
# j = 0
# for i in items_dict:
#     j+=1
#     if items_dict[i]["rated_times"] < threshold:
#         df.drop(df.loc[df["iid"] == i ].index, inplace=True)

# Users con muchos items
users_dict = {}
i = 0
print("Start analyzing users")
# for user in users:
#     # print("user" + str(i)+"/"+str(n_users))
#     users_dict[user] = {}
#     df_aux = df.loc[df['uid']==user].index
#     users_dict[user]["rated_songs"] = df_aux.shape[0]
#     i+=1
# print("Done")
# a_file = open('users_dict.pkl', "wb")
# pickle.dump(users_dict, a_file)
# a_file.close()
a_file = open("users_dict.pkl", "rb")
users_dict = pickle.load(a_file)

for user in users_dict:
    if users_dict[user]["rated_songs"] > 100:
        # print(user, users_dict[user]["rated_songs"])
        pass