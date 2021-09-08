from .algorithm_lib.surprise_model import SurpriseModel

import datetime
from collections import defaultdict

# Facade design pattern: https://en.wikipedia.org/wiki/Facade_pattern
# When the present class is accessed it requests one of the algorithms located in algorithm_lib
# which implement the algorithm_lib/algorithm_base.py interface.
# Cuando se accede a la clase descrita en este archivo esta solicita alguno de los algoritmos situados en algorithm_lib
# Los cuales implementan la interfaz algorithm_lib/algorithm_base.py

"""
    Class in charge of training and executing recommendation algorithms
        Input: correctly loaded user-item-rating matrix
               + Algorithm configurations
 
        Output: Object with the algorithm ready to be used

    Clase encargado del entrenamiento y ejecución de algoritmos de recomendación
        Input: Matriz user-item-rating correctamente cargada
               + Configuraciones algoritmos
 
        Output: Objeto con el algoritmo listo para ser utilizado
"""

class Algorithm():
    def __init__(
        self, dataset,
        lib, algorithm,
        verbose=True
    ):
        """
        Constructor
            :param dataset: Loaded dataset object
            :type DatasetModule: 

            :param lib: lib
            :type string: "surprise"

            :param algorithm: algorithm to be used
            :type dict_key: "svd"
        """

        self.lib_dict = {
            "surprise": SurpriseModel
        }

        self.module = "Algorithm: "

        # User data
        self.dataset = dataset

        self.model = None
        self.lib = lib
        self.algorithm = algorithm

        self.verbose = verbose

        # 1. Comprobamos que el modelo de datos esta correcto para la librería
        if self.dataset_is_correct():
            if self.verbose : print(self.module + "correct dataset.")
        else:
            print(self.module + "error in the dataset object")
            return

        # 2. Se carga el modelo a utilizar
        self.start_model()

        # 3. #TODO: In order to reduce the computational cost of further training, the fitting process could be called here.


    def start_model(self):
        '''
            initializes the model
        '''
        try:
            if self.verbose : print(self.module + "Loading model " + self.algorithm + " from " + self.lib)
            self.model = self.lib_dict[self.lib](self.algorithm, self.dataset)
            if self.verbose : print(self.module + "Loaded model " + self.algorithm + ".")
        except Exception as ex:
            if self.verbose : print(self.module + "No available library has been indicated")
            print(ex)

    def fit_model(self, checktime=True):
        '''
            Fitting the model
        '''
        #TODO comprobación de errores
        start_time = datetime.datetime.now()
        if self.verbose : print(self.module + "Fitting model " + self.algorithm + "...")
        self.model.fit()
        if (checktime and self.verbose) : print(self.module + "Done. Fitting time: ", datetime.datetime.now() - start_time)

    def get_user_prediction(self, user_id, item_id, r_ui=None):
        """ 
            for a given user, returns the predicted rating for an especific item
            :param user_id: user id
            :type string: 

            :param item_id: item id
            :type string: 

            :param r_ui: real rating
            :type string: 
            
            Devuelve valoración 
        """
        #TODO gestionar verbose
        #TODO comprobar si está fit
        # if self.verbose : print(self.module + "Prediction for user " + user_id + " for the item: " + item_id)
        pred = self.model.predict(user_id, item_id, r_ui=r_ui)
        # if self.verbose : print(self.module + "Prediction result: ", pred)
        return pred

    def get_top_n_list(self, user_id, n=10):
        #TODO various of the following functions may be static
        '''
            sets in self.top_n_list the full top-n recomendation for the users
        '''
        #TODO Comprobación fitted
        if self.verbose : print(self.module + "Getting Top-"+str(n)+" lists")
        top_n_list = self.model.get_top_n(n=n)
        #DANGER:
        if self.verbose : print(self.module + "Top-N lists recovered: "+str(len (top_n_list))+" list")
        return top_n_list

    def get_top_n_list_simple_user(self, user_id, n=10):
        '''
            currently almost a copy of get_top_n_list(), a final implementation should
            return only the recommendation for one user without generating it 
            for all others.
        '''
        if self.verbose : print(self.module + "Getting Top-"+str(n)+" list for " + user_id)
        top_n_lists = self.model.get_top_n_list_simple_user(user_id, n=n)
        if self.verbose : print(self.module + "User: " + user_id + " Top-"+str(n)+" list")
        return top_n_lists[int(user_id)]
        
    def print_top_n_list(self, top_n_lists, user_id):
        '''
            Prints the top-n list for a given user
        '''
        print("Top-10 User " + user_id + ": ")
        j = 0
        for x in top_n_lists[int(user_id)]:
            j += 1 
            print("\t"+ str(j)+ ") Item: ", x[0])
        
    def do_cross_validate_model(self, cv=5, checktime=True, verbose=False):
        #TODO: verbose, comprobación errores, implementación distintas librerías
        """
            Performs the cross validation of the model over the individual users
        """
        start_time = datetime.datetime.now()
        if self.verbose : print(self.module + "Cross validating the RS with model " + self.algorithm)
        self.model.cross_validate(cv=cv, verbose=self.verbose)
        if (checktime and self.verbose) : print(self.module + "Tiempo cross validate: ", datetime.datetime.now() - start_time)

    def dataset_is_correct(self):
        #TODO
        """
            Comprueba que el dataset esta inicializado
            que son correctos los campos
            y que coinciden con la librería utilizada
        """
        if self.dataset != None:
            return True
        else:
            return False

    def export_results_to_file(self):
        #TODO
        """
            Exporta resultados de la recomendación a un archivo de texto plano
        """
        return