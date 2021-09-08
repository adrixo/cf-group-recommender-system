
class AlgorithmBase:
    def __init__(self, algorithm: str, dataset_module: str) -> str:
        """
            Carga el modelo y selecciona el algoritmo
        """
        pass

    def start_model(self):
        """
            Inicia y selecciona el modelo con el algoritmo elegido
        """
        pass

    def fit(self):
        """
            Entrena el algoritmo
        """
        pass

    def predict(self, user: str, item: str) -> dict:
        """
            Predice la valoración de un item para un usuario.
            objetos con la forma  (user_id, item_id, r_ui, predicted value)
        """
        pass

    def get_top_n(self, n: int):
        """
            Devuelve un array de listas top-n para los usaurios.
            [(user,[pred, r_uid]) ]
        """
        pass


    def get_top_n_list_simple_user(self, user_id, n=10):
        """
            Predice la valoración de un item para un usuario.
            (user,[pred, r_uid])
        """
        pass

    def cross_validate(self, cv=5, verbose=False):
        """
            Realiza cross validate con los datos del dataframe
        """
        pass