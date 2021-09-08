from .evaluator import Evaluator

"""
    Modulo encargado de la agregación de recomendación
        input: individual algorithm
               Group of users

        output: group recommendation
"""

class Aggregator():
    def __init__(
        self, dataset_module, algorithm_module, group_module,
        lib="surprise", aggregation_method="avg",
        threshold=.6,
        verbose=True, full_verbose=False
    ):
        """
        Constructor
            :param dataset_module: módulo con los datos
            :type dataset_module:
            
            :param algorithm_module: módulo con el algoritmo
            :type algorithm_module:
            
            :param group_module: módulo de grupo
            :type group_module:
            
            :param lib: librería a utilizar
            :type string:
            
            :param aggregation_method: método de agregación
            :type string:
        """

        self.module = "Aggregator: "
        self.item_id = 'iid'

        self.aggregation_methods_dic = { #TODO: add to the tuple a description and full name to make it more readable
            "avg": self.average,
            "add": self.additive_utilitarian,
            "app": self.approval_voting,
            "avm": self.average_without_misery,
            "brc": self.borda_count,
            "cop": self.copeland_rule,
            "fai": self.fairness,
            "lms": self.least_misery,
            "maj": self.majority_voting,
            "mpl": self.most_pleasure,
            "mrp": self.most_respected_person,
            "mul": self.multiplicative,
            "plu": self.plurality_voting,
        }


        self.dataset_module = dataset_module
        self.algorithm_module = algorithm_module
        self.group_module = group_module
        self.lib = lib

        self.threshold = threshold
        self.threshold_value = threshold* self.dataset_module.rating_scale[1]

        self.evaluator = Evaluator(lib=self.lib)

        self.aggregation_function = self.aggregation_methods_dic[aggregation_method]
        self.aggregation_method = aggregation_method

        self.verbose = verbose
        self.full_verbose = full_verbose

        self.group_recommendation_list = None 

    def perform_group_recommendation(self): # Algunos métodos de agregación no serían generalizables de esta forma
        """
            1. Para cada item de la lista
                2. Para cada usuario del grupo
                    3. Se coge la predicción para ese item 
                2.2 Se agregan las predicciones en una única predicción

        """
        aggregated_items = {
            #item: aggregated_rating
        }
        for item in self.dataset_module.df[self.item_id].unique().tolist(): #TODO: warning with artistID
            item_ratings = []
            try:
                for user in self.group_module.users:
                    # Primero se intenta obtener la predicción real?

                    # Si no existe se realiza la predicción
                    rating = self.algorithm_module.get_user_prediction(user, item)
                    ra = rating[3]
                    item_ratings.append(ra)
            except Exception as ex:
                pass
            
            aggregated_items[item] = self.aggregation_function(item_ratings)

        aggregated_items = self.sort_dict_by_value(aggregated_items)
        
        self.group_recommendation_list = aggregated_items 
    
        return aggregated_items

    def sort_dict_by_value(self, dictionary):
        return dict(reversed(sorted(dictionary.items(), key=lambda kv: kv[1])))

    def print_group_recommendation(self):
        if self.group_recommendation_list == None:
            print(self.module + " Todavía no se ha realizado la recomendación. ")
        else:
            print(self.module + "\nGroup name: " + self.group_module.group_name + " - context: " + self.group_module.group_context_name)
            print("Top-N list using " +self.aggregation_method+ " aggregation method:")
            i = 0
            for item in self.group_recommendation_list:
                i+=1
                print("\t"+str(i)+") Song: ", item, " Rating: ", self.group_recommendation_list[item])

    def evaluate(self, threshold=0.6, k=(10,11,10)):
        if self.verbose : print(self.module + "Evaluating group prediction "+self.aggregation_method+"@"+str(k)+": ")
        metrics = []
        metrics += self.evaluator.get_precision_recall(self.group_recommendation_list, self.group_module, self.dataset_module, self.algorithm_module, self.aggregation_function, threshold=threshold, k=k)
        # metrics += self.evaluator.get_mae_rsme(self.group_recommendation_list, self.group_module, self.dataset_module, self.algorithm_module, self.aggregation_function, threshold=threshold, k=k)
        metrics += self.evaluator.get_dcg_ndcg(self.group_recommendation_list, self.group_module, self.dataset_module, self.algorithm_module, self.aggregation_function, threshold=threshold, k=k)
        # serendipity, unknown_usefull_recommendations = self.evaluator.get_serendipity(self.group_recommendation_list, self.group_module, self.dataset_module, self.algorithm_module, self.aggregation_function, threshold=threshold, k=k[0])
        # coverage = self.evaluator.get_coverage(self.group_recommendation_list, self.group_module, self.dataset_module, self.algorithm_module, self.aggregation_function, threshold=threshold, k=k[0])
        # consensus, fairness = self.evaluator.get_consensus_fairness(self.group_recommendation_list, self.group_module, self.dataset_module, self.algorithm_module, self.aggregation_function, threshold=threshold, k=k[0])
        return metrics

# Aggregation functions 

    #Consensus-based
    def average(self, rating_list):
        """ 
            Media valoraciones item
        """
        avg = 0
        for rating in rating_list:
            avg += rating

        if len(rating_list) == 0:
            return 0

        avg = avg / len(rating_list)
        return avg
        
    #Consensus-based
    def additive_utilitarian(self, rating_list):
        """ 
            Suma valoraciones item
        """
        add = 0
        for rating in rating_list:
            add += rating
        return add
                
    #Consensus-based
    def multiplicative(self, rating_list):
        """ 
            Multiplicación valoraciones item
        """
        mul = 1
        for rating in rating_list:
            mul *= rating
        return mul

    #Consensus-based
    def average_without_misery(self, rating_list):
        """ 
            Media de las valoraciones que superen cierto umbral
        """
        #TODO: Comprobar división por 0
        avm = 0
        avm_tot = 0
        for rating in rating_list:
            if rating >= self.threshold_value:
                avm += rating
                avm_tot += 1

        if avm_tot == 0:
            return 1
        avm = avm / avm_tot
        return avm

    # Consensus based
    def fairness(self, rating_list): # requiere iteraciones y distinta implementación a la iterativa tradicional
        return -1

    #Majority-based
    def approval_voting(self, rating_list): #threshold
        """ 
            Si supera un cierto threshold se añade voto
        """
        app = 0
        for rating in rating_list:
            if rating >= self.threshold_value:
                app += 1
        return app
        
    # Majority-based
    def plurality_voting(self, rating_list):
        """ 
            Item con los maximas votaciones para cada integrante
        """
        plu = 0
        return plu
    
    # Majority-based
    def copeland_rule(self, rating_list):
        # Las victorias suman 1
        # Las derrotas restan 1
        # Deducir la comparativa derrota
        return -1
         
    # Majority-based   
    def borda_count(self, rating_list): # Require de ordenación de los items de cada usuario para poder hacer suma
        # Se suma por posición n-0 ordenada entre todos los votos
        # Si coincide se da el medio punto para cada uno
        return -1

    # Borderline aggregation
    def least_misery(self, rating_list):
        """
            Minimun item-specific evaluation
        """
        min = 1000
        for rating in rating_list:
            if min >= rating:
                min = rating
        return min
                
    # Borderline aggregation
    def majority_voting(self, rating_list): # Podría requerir cierta normalización, se propone redondeo a 0.5
        """
            Majority of evaluation values per item
        """
        majority_dict = {
            # value: num_users_with_that_vote
        }
        for rating in rating_list:
            rounded_rating = round(rating)
            if rounded_rating in majority_dict:
                majority_dict[rounded_rating] += 1
            else:
                majority_dict[rounded_rating] = 0
        most_voted = max(majority_dict, key=majority_dict.get)
        return most_voted

    # Borderline aggregation
    def most_pleasure(self, rating_list):
        """
            Maximum item-specific evaluation
        """
        max = -1000
        for rating in rating_list:
            if max <= rating:
                max = rating
        return max

    # Borderline aggregation
    def most_respected_person(self, rating_list):
        return -1


