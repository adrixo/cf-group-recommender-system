# This script processes datasets, either to reduce their size by eliminating columns, 
# or to obtain valid identifiers (integers) from text strings using hash functions 

from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import hashlib

# Hash function to obtain the numeric id
def simpleHash(string):
    x = int(hashlib.sha1(string.encode("utf-8")).hexdigest(), 16) % (10 ** 10)
    return x

dataset = 'dataset.csv'
output_file = "output_dataset.csv"
train_file_output = "train.csv"
test_file_output = "test.csv"
header = True
sep = ','

df = pd.read_csv(dataset, sep=sep)

# Column elimination
# del df['n']
# del df['plays']

df["uid"] = NaN
df["iid"] = NaN
for i in df.index: 
    df["uid"][i] = simpleHash(df["user"][i])
    df["iid"][i] = simpleHash(df["spotify_track_id"][i])

df['uid'] = df['uid'].astype(str)          
df['uid'] = df['uid'].str.replace('.0', '')
df['iid'] = df['iid'].astype(str)
df['iid'] = df['iid'].str.replace('.0', '')

columns = ["uid", "iid", "rating", "user", "track_name", "artist_name", "play_count_track", 
            "play_count_total_user", "spotify_track_id", "explicit", "danceability", "energy", "key", 
            "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", 
            "tempo", "duration_ms"]
df = df[columns]

df.to_csv(dataset, index=False, header=header, sep='\t')
# train, test = train_test_split(df, test_size=0.2)
# train.to_csv(train_file_output, index=False, header=header, sep="\t")
# test.to_csv(test_file_output, index=False, header=header, sep="\t")
