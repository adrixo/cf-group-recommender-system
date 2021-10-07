import sys
sys.path.append("../../")
from random import random
from random import randint
from random import seed
seed(1)

from configuration import *
import pandas as pd

dataset_file = '../../'+cfg['experiment_dataset']
output_file_unified = '../experiment_groups.py'
sep = cfg['sep']

f = open(output_file_unified, "w")
f.write('')
f.close()

threshold = cfg['relevant_threshold']
rating_scale = cfg['rating_scale']
threshold_value = threshold*rating_scale[1]

n_groups = 50
group_size = 5
context_relevant_songs_threshold = 5
min_relevant_ratings = 20

print("loading " + dataset_file)
df = pd.read_csv(dataset_file, sep=sep)
print(df)
print(df.shape)
# All users
users = df["uid"].unique().tolist()
n_users = len(users)-1
print("Number of users: ", len(users))

##################
# 1. Random groups

# Random groups array of arrays
random_groups = []
# Array of repeated users to delete
to_delete = []#{group:i, member:j}
# 1.1 creation of random groups without considering the context
for n_group in range(n_groups):
    random_groups.append([])
    for user in range(group_size):
        user_pos = randint(0, n_users-1)
        user_id = users[user_pos]
        random_groups[n_group].append(user_id)

# 1.2 Check for repeated users
for n_group in range(n_groups):
    for i in range(len(random_groups[n_group])):
        for j in range(i+1, len(random_groups[n_group])):
            if random_groups[n_group][i] == random_groups[n_group][j]:
                to_delete.append({'group': n_group, 'member': j})
# 1.3 deletion of repeated users
for d in to_delete[::-1]:
    random_groups[d["group"]].pop(d["member"])

# 1.4 groups are saved in the format [{"name":"test1","members":["1","2"]]}]
group_prefix = "random_"
all_groups_list_string = "["
for i in range(len(random_groups)):
    group = random_groups[i]
    group_name = group_prefix + str(i)
    members_string = "["
    for member in group:
        members_string += "'" + str(member) + "',"
    members_string = members_string[:-1]
    members_string += "]"
    group_object = "{'name':'"+group_name+"','members':"+members_string+"}"
    all_groups_list_string += group_object + ","
all_groups_list_string = all_groups_list_string[:-1]
all_groups_list_string += "]"

print("Saving random groups...")
# 1.5 group saving into 'output_file_unified' file
all_groups_list_string = group_prefix[:-1] + " = "+ all_groups_list_string
f = open(output_file_unified, "a")
f.write(all_groups_list_string)
f.write("\n")
f.close()
print("Random groups saved")

# Given a threshold, returns the user's number of positive ratings 
def get_user_all_relevant_items(user, threshold_value, df):
    try: 
        user_columns = df.loc[df['uid'] == int(user)]
        relevant_items_rows = user_columns.loc[user_columns['rating'] >= threshold_value]
        relevant_items = relevant_items_rows.shape[0]
    except Exception as e:
        relevant_items = 0
    return relevant_items

######################
# 2. Contextual groups
# Several filters are applied:
#   1. context filter e.g. chill context filter leaves only those records with energy < 10% 
#   2. non-relevant records filtering: only those records whose rating exceeds a certain threshold
#   3. songs rated for at least a certain number of users
#   4. user minimal relevant songs filtering. a user must have rated a minimum number of songs positively in that context
#   5. 

contexts_colection_def = [ cfg['chill_context'], cfg['fitness_context'], cfg['party_context']]
# 2.1. For each filter, we filter the dataset and keep only those songs related to the context
for f in contexts_colection_def:

    print("")
    fdf = df.copy() # filtered dataframe

    # 2.2 song filtering
    for filter in f['context']: # Can have several filters
        if filter['direction'] == "above":
            fdf.drop(fdf.loc[fdf[filter['column']]<filter['threshold']].index, inplace=True)
        if filter['direction'] == "below":
            fdf.drop(fdf.loc[fdf[filter['column']]>filter['threshold']].index, inplace=True)

    # At this point in fdf we would have only those user-item-rating records with contextual items
    #users = df["uid"].unique().tolist() # Only those users with at least one song in context

    # 2.3 non-relevant records filtering: At this point in fdf we would have only those records with relevant ratings
    fdf.drop(fdf.loc[fdf['rating']<threshold_value].index, inplace=True)
    print("FILTER - ", f)

    # 2.4 We can also keep only those songs that have a certain number of ratings
    items = fdf["iid"].unique().tolist()
    users = fdf["uid"].unique().tolist()
    print("# songs: ", len(items))
    print("# users: ", len(users))
    for item in items:
        item_index = fdf.loc[fdf['iid']==item].index
        if item_index.shape[0] < context_relevant_songs_threshold: # If there is no minimum number of ratings for the item
            fdf.drop(item_index, inplace=True)
    print("Result after removing non-relevant items:")  
    items = fdf["iid"].unique().tolist()
    users = fdf["uid"].unique().tolist()
    print("# songs: ", len(items))
    print("# users: ", len(users))

    # 2.5 user minimal relevant songs filtering: Here we would already have users contextual items with at least context_relevant_songs_threshold songs
    user_aux = []
    for user in users:
        relevant_ratings = get_user_all_relevant_items(user, threshold_value, fdf)
        if relevant_ratings > min_relevant_ratings:
            user_aux.append(user)
    users = user_aux
    n_users = len(users)
    print("Result after removing users with less relevant songs than ", min_relevant_ratings)
    print("# of eligible users: ", len(users))

    context_groups = []
    # 2.6 creation of random groups considering the context
    to_delete = []#{group:i, member:j}
    for n_group in range(n_groups):
        context_groups.append([])
        for user in range(group_size):
            user_pos = randint(0, n_users-1)
            user_id = users[user_pos]
            context_groups[n_group].append(user_id)

    # 2.7 Check for repeated users and deletion of repeated users
    for n_group in range(n_groups):
        for i in range(len(context_groups[n_group])):
            for j in range(i+1, len(context_groups[n_group])):
                if context_groups[n_group][i] == context_groups[n_group][j]:
                    to_delete.append({'group': n_group, 'member': len(context_groups[n_group])-j-1})
    for d in to_delete[::-1]:
        context_groups[d["group"]].pop(d["member"])

    # 2.8 groups are saved in the format [{"name":"test1","members":["1","2"]]}]
    group_prefix = f['name'] + "_"
    all_groups_list_string = "["
    for i in range(len(context_groups)):
        group = context_groups[i]
        group_name = group_prefix + str(i)
        members_string = "["
        for member in group:
            members_string += "'" + str(member) + "',"
        members_string = members_string[:-1]
        members_string += "]"
        group_object = "{'name':'"+group_name+"','members':"+members_string+"}"
        all_groups_list_string += group_object + ","
    all_groups_list_string = all_groups_list_string[:-1]
    all_groups_list_string += "]"

    # 2.9 group saving into 'output_file_unified' file
    all_groups_list_string = group_prefix[:-1] + " = "+ all_groups_list_string
    f = open(output_file_unified, "a")
    f.write(all_groups_list_string)
    f.write("\n")
    f.close()
