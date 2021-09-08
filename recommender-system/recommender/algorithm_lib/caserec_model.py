from .algorithm_base import AlgorithmBase

class CaserecModel(AlgorithmBase):

    def __init__(self, algorithm, dataset_module):
        AlgorithmBase.__init__(self, algorithm, dataset_module)

    def start_model(self):
        pass

    def fit(self):
        pass

    def predict(self, user_id, item_id, r_ui=None) -> dict:
        pass

    def get_top_n(self, n=10):
        pass

    def get_top_n_list_simple_user(self, user_id, n=10):
        pass

    def parse_surprise_n_predictions(self, predictions, n=10):
        pass

    def cross_validate(self, cv=5, verbose=False):
        pass