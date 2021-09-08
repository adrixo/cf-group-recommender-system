from math import sqrt
from math import log

"""
    Module in charge of the evaluation of a recommendation for groups.
    IMPORTANT: in order to make substitutable evaluation functions, they all have the same arguments

    Current developed metrics: 
     - dcg, ndcg
     - mae, rsme
     - precision, recall
    
    Other unfinished proposals:
     - serendipity
     - coverage
     - consensus, fairness
"""

class Evaluator():
    def __init__(
        self, lib="surprise",
        verbose=True
    ):
        """
        Constructor
            :param lib: Librería utilizada
            :type string: "surprise"
        """
        self.module = "Evaluator: "
        self.item_id_name = "iid"
        self.user_id_name = "uid"

        self.lib = lib

    # Information used in recsys evaluation:
        # Users real ratings: pandas dataframe in the dataset object 
        # Estimated individual values: obtained through the algorithm 
        # Estimated ratings for the group: aggregator
        # Group real values: aggregation method

        self.verbose = verbose

    # Se propone utilizar el valor del artículo real, en caso de no existir: ninguno
    def get_user_real_rating_from_df(self, user, item, df):
        '''
            Returns, if available, the real rating of an item for a given user.
            Returns none if the user doesnt have a real evaluation 
        '''
        try: 
            user_columns = df.loc[df[self.user_id_name] == int(user)] #TODO normalizar tipos
            item_row = user_columns.loc[user_columns[self.item_id_name] == int(item)] #TODO: Mucho cuidado con el contexto, podŕia haber elementos repetidos en distintos contextos
            real_rating = item_row.iat[0,2] #TODO: normalizar que esté el 3o o acceder por nombre normalizado

        except Exception as e:
            # print("get_user_real_rating_from_df: No prediction for user", e)
            real_rating = None
        return real_rating

    # TODO: comprobar k list
    def get_precision_recall(self, group_recommendation_list, group, dataset, algorithm, aggregation_function, k=10, threshold=.60, recall_depth=100):
        """
            Precision: Number of relevant elements recommended / total number of elements recommended.
                relevant elements recommended:

            Precisión: Numero de elementos relevantes recomendados / numero total de elementos recomendados
                elementos relevantes recomendados:
        """
        try:
            threshold_value = threshold * dataset.rating_scale[1]
            
            real_relevant_items = 0
            predicted_relevant_items = 0
            true_positives = 0
            total_items = 0

            #Variables de depuración
            real_ratings = 0
            predicted_ratings = 0

            group_members_data = {}
            for user in group.users:
                group_members_data[user] = {
                    "true_positives": 0,
                    "predicted_relevant_items": 0,
                    "real_relevant_items": 0
                }

            for group_item in group_recommendation_list:
                
                if total_items > k:
                    break

                total_items += 1
                
                group_item_predicted_rating = group_recommendation_list[group_item]
                #print("item;", group_item, "rating", group_item_predicted_rating)

                users_real_ratings = []
                # Se obtienen los valores de los ratings de los usuarios para calcular el aggregado real
                for user in group.users:
                    # Seleccion valor real valoración de un item
                    real_rating = self.get_user_real_rating_from_df(user, group_item, dataset.df)

                    # Si para un usuario no existiese una valoración real se toma la predicción
                    if real_rating == None:
                        real_rating = algorithm.get_user_prediction(user, group_item)[3] #TODO normalizar el [3]
                        predicted_ratings += 1
                    else:
                        real_ratings += 1
                    #print("useritemvsg",user, group_item, real_rating, "g", group_item_predicted_rating)
                    
                    users_real_ratings.append(real_rating)
                    #1. Metrica precisión comparación de valoraciones reales primero
                    is_real_relevant = False
                    is_predicted_relevant = False

                    if real_rating > threshold_value:
                        is_real_relevant = True
                        group_members_data[user]["real_relevant_items"] += 1
                        
                    if group_item_predicted_rating > threshold_value:
                        is_predicted_relevant = True
                        group_members_data[user]["predicted_relevant_items"] += 1

                    is_correctly_classified = is_real_relevant and is_predicted_relevant
                    if is_correctly_classified:
                        group_members_data[user]["true_positives"] += 1

                #2. Metrica precisión agregación de valoraciones reales primero
                group_real_rating = aggregation_function(users_real_ratings)

                is_real_relevant = False
                is_predicted_relevant = False

                if group_real_rating > threshold_value:
                    is_real_relevant = True
                    real_relevant_items += 1
                if group_item_predicted_rating > threshold_value:
                    is_predicted_relevant = True
                    predicted_relevant_items += 1

                is_correctly_classified = is_real_relevant and is_predicted_relevant
                if is_correctly_classified:
                    true_positives += 1


            # print("chekcing", real_ratings, predicted_ratings, real_ratings/4, predicted_ratings/4)
                # print("check", group_item, group_real_rating, group_item_predicted_rating, is_real_relevant, is_predicted_relevant, is_correctly_classified)
            #1. metrica media users
            all_precisions = []
            all_recalls = []
            for user in group.users:
                precision_u, recall_u = self.calc_precision_recall(total_items, group_members_data[user]["real_relevant_items"], group_members_data[user]["predicted_relevant_items"] )
                #precision_u, recall_u = self.calc_precision_recall(total_items, group_members_data[user]["true_positives"], group_members_data[user]["predicted_relevant_items"] )
                print("user "+user+" p and c @"+str()+": ")
                all_precisions.append(precision_u)
                all_recalls.append(recall_u)
            
            precision_2 = self.calc_list_average(all_precisions)
            recall_2 = self.calc_list_average(all_recalls)

            #2. metrica valoriones reales primero
            precision, recall = self.calc_precision_recall(total_items, real_relevant_items, predicted_relevant_items)
            #precision, recall = self.calc_precision_recall(total_items, true_positives, predicted_relevant_items)

            # print("INFO: precall", total_items,"tp", true_positives,"rr", real_relevant_items, "pr", predicted_relevant_items)
            # print("INFO: Precision: ", precision, true_positives, total_items)
            # print("INFO: recall: ", recall, true_positives, predicted_relevant_items)

            # print("Precision and recall @"+str(k)+":", precision, recall)
            print("Precision and recall v2: @"+str(k)+":", precision_2, recall_2)
            
            return precision, recall, precision_2, recall_2
        except Exception as ex:
            print("get_precision_recall ex: ", ex)
            return "err", "err"
    
    def calc_precision_recall(self, total_items, true_positives, relevant_items):
        '''
            Given the amount of items in a top-n recomendation (total_items) the relevant items (true_positives) and
            all the relevant items for a certain user (relevant items), the precision and the recall are calculated and returned.
        '''
        if total_items == 0:
            print("total_items: TREMENDO WARNING: DIVISION POR CERO PRECISION")
            total_items = -1

        precision = true_positives / total_items

        if (relevant_items == 0):
            print("relevant_items: TREMENDO WARNING: DIVISION POR CERO PRECISION")
            relevant_items = -1

        recall = true_positives / relevant_items
        return precision, recall

    def calc_list_average(self, array):
        '''
            given a number array, returns the average
        '''
        if len(array) == 0:
            return 0

        total = 0
        for n in array:
            total += n
        return total / len(array)

    def get_dcg_ndcgget_mae_rsme(self, group_recommendation_list, group, dataset, algorithm, aggregation_function, threshold=0.6, k=10):
        """

            En recomendación para grupos también se puede considerar la diferencia entre items relevantes y predichos
        """
        try:
            error_sum = 0
            error_squared_sum = 0

            total_items = 0

            #Variables de depuración

            group_members_data = {}
            for user in group.users:
                group_members_data[user] = {
                    "error_sum": 0,
                    "error_squared_sum": 0
                }

            for group_item in group_recommendation_list:
                total_items += 1
                if total_items > k:
                    break

                group_item_predicted_rating = group_recommendation_list[group_item]

                users_real_ratings = []
                # Se obtienen los valores de los ratings de los usuarios para calcular el aggregado real
                for user in group.users:
                    # Seleccion valor real valoración de un item
                    real_rating = self.get_user_real_rating_from_df(user, group_item, dataset.df)

                    # Si para un usuario no existiese una valoración real se toma la predicción
                    if real_rating == None:
                        real_rating = algorithm.get_user_prediction(user, group_item)[3] #TODO normalizar el [3]
                        real_rating = 0.5
                    users_real_ratings.append(real_rating)

                    group_member_error = abs(real_rating - group_item_predicted_rating)
                    group_members_data[user]["error_sum"] += group_member_error 
                    group_members_data[user]["error_squared_sum"] += group_member_error**2

                group_real_rating = aggregation_function(users_real_ratings) #ADRIAN... SI YA LO TENIAS APUNTADO! Problema: cuando todos son inferidos... el error es 0 porque no tenemos suficientes datos ciertos
                group_item_error = abs(group_real_rating - group_item_predicted_rating)
                error_sum += group_item_error
                error_squared_sum += group_item_error**2


            #TODO comprobación errores
            mae_2 = group_members_data[user]["error_sum"] /total_items
            rmse_2 = sqrt(group_members_data[user]["error_squared_sum"] /total_items)

            mae = error_sum/total_items
            rmse = sqrt(error_squared_sum/total_items)

            print("MAE y RMSE: @"+str(k)+":", mae, rmse)
            print("MAE y RMSE v2: @"+str(k)+":", mae_2, rmse_2)

            return mae, rmse, mae_2, rmse_2
        except Exception as ex:
            print("get_mae_rsme ex: ", ex)
            return "err", "err"

    #TODO: Se pueden dejar todos los mismos valores en todas las métricas para homogeneizarlo o quitar las necesarias
    def get_dcg_ndcg(self, group_recommendation_list, group, dataset, algorithm, aggregation_function, threshold=0.6, k=10):
        try:
            
            threshold_value = threshold * dataset.rating_scale[1]

            total_items = -1
            dcg = 0
            idcg = 0

            for group_item in group_recommendation_list:
                if total_items > k:
                    break
                total_items += 1

                if group_recommendation_list[group_item] >= threshold_value:
                    relevance = 1
                else:
                    relevance = 0

                dcg += 2**(relevance - 1) / log(1+total_items, 2)
                idcg += 1 / log(1+total_items, 2)

            ndcg = dcg/idcg

            print("DCG y nDCG @"+str(k)+":", dcg, ndcg)

            return dcg, ndcg
        except Exception as ex:
            print("get_dcg_ndcg ex: ", ex)
            return "err", "err"

    def get_serendipity(self, group_recommendation_list, group, dataset, algorithm, aggregation_function, k=10, threshold=.60):
        #TODO 
        """
            Como no se va a comparar entre dos algoritmos en esta función, se propone como serendipia:
                La proporcion de elementos relevances desconcoidos para un grupo en relacion con los relevantes conocidos
                Entenidiendo como elevante desconocido aquel que se ha genereado por predicción.
        """
        try:
            threshold_value = threshold * dataset.rating_scale[1]

            total_items = 0

            known_usefull_recommendations = 0
            unknown_usefull_recommendations = 0

            for group_item in group_recommendation_list:
                if total_items > k:
                    break
                total_items += 1

                for user in group.users:
                    real_rating = self.get_user_real_rating_from_df(user, group_item, dataset.df)

                    if real_rating == None:
                        if unknown_usefull_recommendations >= threshold_value:
                            unknown_usefull_recommendations += 1
                    else:
                        if known_usefull_recommendations >= threshold_value:
                            known_usefull_recommendations += 1

            if total_items == 0:
                print("TREMENDO WARNING: DIVISION POR CERO PRECISION")
                total_items = 1

            serendipity = unknown_usefull_recommendations / total_items
            print("Serendipity, unkown @"+str(k)+":", serendipity, unknown_usefull_recommendations)
            
            return serendipity, unknown_usefull_recommendations
        except Exception as ex:
            print("get_serendipity ex: ", ex)
            return "err", "err"


    def get_serendipityget_coverage(self, group_recommendation_list, group, dataset, algorithm, aggregation_function, k=10, threshold=.60):
        '''
            #TODO
        '''
        try:
            return 0
        except Exception as ex:
            print("get_coverage ex: ", ex)
            return "err", "err"

    def get_consensus_fairness(self, group_recommendation_list, group, dataset, algorithm, aggregation_function, k=10, threshold=.60):
        '''
            #TODO
        '''
        try:
            consensus = 0
            fairness = 0
            return consensus, fairness
        except Exception as ex:
            print("get_consensus_fairness ex: ", ex)
            return "err", "err"
