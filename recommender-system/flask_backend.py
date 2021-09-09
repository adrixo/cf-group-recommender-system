

from recommender.dataset import Dataset
from recommender.algorithm import Algorithm
from recommender.group import Group
from recommender.aggregator import Aggregator

from server_configuration import *
import pandas as pd

import hashlib
def simpleHash(string):
    x = int(hashlib.sha1(string.encode("utf-8")).hexdigest(), 16) % (10 ** 10)
    return x

import threading
import time


from flask import Response, request, Flask, send_from_directory, jsonify
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse

from pathlib import Path
currentPath = Path(__file__).resolve().parent

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
cred = credentials.Certificate('./*.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


# Variables de configuración
dataset_file = "datasets/api_dataset.csv"
sep = cfg['sep']
columns = cfg['columns']
rating_scale = cfg['rating_scale']
lib = cfg['lib']
verbose = cfg['verbose']
algorithm = cfg['algorithm']

# Proceso de carga del programa
def generateUserRegister(user, song_params, value):
    args = [user, simpleHash(song_params["features"]["id"]), value, user]
    args += [song_params["track"]["artists"][0]["name"], 10, 10]
    args += [song_params["features"]["id"]]
    args += [song_params["features"]["id"], 'False']
    args.append(song_params["features"]["danceability"])
    args.append(song_params["features"]["energy"])
    args.append(song_params["features"]["key"])
    args.append(song_params["features"]["loudness"])
    args.append(song_params["features"]["mode"])
    args.append(song_params["features"]["speechiness"])
    args.append(song_params["features"]["acousticness"])
    args.append(song_params["features"]["instrumentalness"])
    args.append(song_params["features"]["liveness"])
    args.append(song_params["features"]["valence"])
    args.append(song_params["features"]["tempo"])
    args.append(song_params["features"]["duration_ms"])
    line = ""
    for arg in args:
        line += str(arg) + "\t"
    line = line[:-1]
    return line

def save_new_rating(rating_line):
    global dataset_file
    f = open(dataset_file, "a")
    f.write(rating_line + "\n")
    f.close()


def user_has_rating(user, song):
    global dataset_file
    df = pd.read_csv(dataset_file, sep='\t')
    real_rating = True
    try: 
        user_columns = df.loc[df['uid'] == int(user)]
        item_row = user_columns.loc[user_columns['iid'] == int(simpleHash(song))]
        real_rating = item_row.iat[0,2]
    except Exception as e:
        # Se obtiene una valoración baja = no será relevante
        real_rating = False
    return real_rating

app = Flask(__name__)
CORS(app)
api = Api(app)

@app.route("/group_recommendation", methods=["GET", 'POST'])
def group_recommendation():
    groupid = request.json['group']
    response = {
        'error': '0',
        'top_n': '0',
        'code': 0,
        'status': 200
    }
    print("asdfgroup_recommendation")
    try:
        print("Getting users from db: ")
        print(request.json['group'])
        group_name = request.json['group']
        users_collections = db.collection(u'rooms').document(groupid).collection(u'users').stream()
        users = []
        for doc in users_collections:
            users.append(doc.to_dict())

        # 1. Se guardan las canciones para todos los usuarios en el dataset
        users_group = []
        for user in users:
            users_songs = user['songs']['items']
            for song in users_songs:
                if not user_has_rating(user['id'], song["features"]["id"]):
                    print("user doesnt has the rating, adding")
                    line = generateUserRegister(user['id'], song, 4)
                    save_new_rating(line)
                    db.collection(u'songs').document(str(simpleHash(song["features"]["id"]))).set(song)

            users_group.append(user['id'])


        # 2. Se genera la predicción para el grupo
        # Se carga el dataset y se añaden los usuarios
        dm = Dataset(dataset_file, 
                        sep='\t', 
                        rating_scale=(0,4), 
                        cols=["uid", "iid", "rating"], 
                        split_dataset=True, 
                        lib='surprise', 
                        test_size=0.2,
                        line_format='user item rating',
                        verbose=True)

        # 3. Se almacena la predicción en la cola
        # TODO

        # # 3. Se genera el grupo
        group = Group(group_name=group_name, group_context_name="web")
        group.add_list_of_users(users_group)

        # # 4. Se entrena el modelo 
        am = Algorithm(dm, lib=lib, algorithm=algorithm)
        am.fit_model()

        # # 5. Se realiza la agregación
        aggregation = Aggregator(dm, am, group, aggregation_method="mpl")
        rec = aggregation.perform_group_recommendation()

        aggregation.print_group_recommendation()


        # 6. Obtenemos la playlist en formato correcto;
        queue = []
        n_songs = 0
        for song_id in rec:
            if n_songs >=20:
                break
            n_songs += 1
            new_song = db.collection(u'songs').document(str(song_id)).get()
            if new_song.exists:
                queue.append(new_song.to_dict())
            else:
                print(u'No such song!', song_id)

        # # 7. Opcionalmente se evalua el resultado
        # aggregation.evaluate()

        # Se sube la playlist a la base datos, buscando forzar la reactividad en el prototipo
        db.collection(u'rooms').document(groupid).collection(u'playlists').document('queue').set({"items": queue})

        response['code'] = 200

    except Exception as ex:
        print("error: ", ex)
        response['code'] = 404
        
    return Response(response)


# Función guarda si una canción ha sido valorada positivamente o negativamente
@app.route("/like_song", methods=["GET", 'POST'])
def like_song():
    response = {
        'error': '0',
        'top_n': '0',
        'code': 0,
        'status': 200
    }
    try:
        song_params = request.json['song_params']
        user = request.json['user']
        value = request.json['value']

        if not user_has_rating(user, song_params["features"]["id"]):
            print("user doesnt has the rating, adding")
            line = generateUserRegister(user, song_params, value)
            save_new_rating(line)
        else:
            print("user has de rating, ignoring")
        
        return Response(response)
    except Exception as ex:
        print("error: ", ex)
        response['code'] = 404
        
    return Response(response)



if __name__ == "__main__": 
    # thread = Function(0.5) # Si hiciese falta threading
    # thread.start()
    app.run( port=5000, debug=True, use_reloader=False) # host="0.0.0.0",

