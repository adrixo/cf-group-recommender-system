from .algorithm_base import AlgorithmBase
from surprise import SVD
from surprise import KNNBasic
from surprise import SlopeOne
from surprise import BaselineOnly
from surprise import CoClustering
from surprise import NMF
from surprise.model_selection import cross_validate

# Surprise documentation:
# https://surprise.readthedocs.io/en/stable/ 


from collections import defaultdict

class SurpriseModel(AlgorithmBase):

    def __init__(self, algorithm, dataset_module):
        AlgorithmBase.__init__(self, algorithm, dataset_module)

        self.surprise_models_dic = {
            "svd": SVD,
            "knn": KNNBasic,
            "nmf": NMF,
            "slopeone": SlopeOne,
            "coclustering": CoClustering,
            "baseline": BaselineOnly
        }

        # Modelo a cargar, por ejemplo SVD, KNNBasic()...
        self.model = None
        self.algorithm = algorithm

        self.dataset_module = dataset_module

        self.start_model()


    def start_model(self):
        self.model = self.surprise_models_dic[self.algorithm]()
        pass

    def fit(self):
        """Entrena el algoritmo"""
        #TODO comprobación si existe train/testset
        self.model.fit(self.dataset_module.trainset)
        pass

    def predict(self, user_id, item_id, r_ui=None) -> dict:
        """Predice la valoración de un item para un usuario."""
        #TODO gestionar verbose
        #TODO comprobar si está fit
        # Comprobar siempre si son ints
        user_id = int(user_id)
        item_id = int(item_id)
        pred = self.model.predict(user_id, item_id, r_ui, verbose=False)
        pred = (user_id, item_id, r_ui, pred[3])
        return pred

    def get_top_n(self, n=10):
        #TODO sería estática
        #TODO Comprobación fitted
        #TODO comprobación testset
        predictions = self.model.test(self.dataset_module.testset)
        top_n_list = self.parse_surprise_n_predictions(predictions, n)
        return top_n_list

    def get_top_n_list_simple_user(self, user_id, n=10):
        #TODO sería estática
        #TODO Comprobación fitted
        #TODO comprobación testset
        #TODO probablemente ineficiente parseo y sea mejor usar siempre get_top_n
        predictions = self.model.test(self.dataset_module.testset)
        top_n_list = self.parse_surprise_n_predictions(predictions, n)
        return top_n_list

    def parse_surprise_n_predictions(self, predictions, n=10):
        top_n = defaultdict(list)

        # Se crea el array de predicciones
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, [est,true_r]))

        # Se ordena y se devuelven las últimas n canciones
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1][0], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n


    def cross_validate(self, cv=5, verbose=False):
        """"""
        #TODO: devuelve las métricas
        cross_validate(self.model, self.dataset_module.data, measures=['RMSE', 'MAE'], cv=cv, verbose=verbose)