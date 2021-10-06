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
        verbose=True, full_verbose=False
    ):
        """
        Constructor
            :param lib: Librería utilizada
            :type string: "surprise"
        """
        self.module = "EvaluationModule: "
        self.item_id = "iid"
        self.user_id = "uid"

        self.lib = lib

        # Information used in recsys evaluation:
        # Users real ratings: pandas dataframe in the dataset object 
        # Estimated individual values: obtained through the algorithm 
        # Estimated ratings for the group: aggregator
        # Group real values: aggregation method

        self.verbose = verbose

    # Real rating from the user, if none is considered not relevant
    def get_user_real_rating_from_df(self, user, item, df):
        try:
            user_columns = df.loc[df[self.user_id] == int(user)] #TODO normalizar tipos
            item_row = user_columns.loc[user_columns[self.item_id] == int(item)] #TODO: Mucho cuidado con el contexto, podria haber elementos repetidos en distintos contextos
            real_rating = item_row.iat[0,2] #TODO: normalizar que esté el 3o o acceder por nombre normalizado

        except Exception as e:
            # print("get_user_real_rating_from_df: No prediction for user", e)
            real_rating = 1
        return real_rating

    def get_user_all_relevant_items(self, user, threshold_value, df):
        try: 
            user_columns = df.loc[df[self.user_id] == int(user)]
            relevant_items_rows = user_columns.loc[user_columns['rating'] >= threshold_value]
            relevant_items = relevant_items_rows.shape[0]
        except Exception as e:
            relevant_items = 0
        return relevant_items

    def create_metric_dict(self, k, name, value):
        return {"k":k, "metric_name": name, "metric_value": value}

    # TODO: comprobar k list
    # TODO:NOW pasar k=(inicio, final, saltos) y se devuelve un array con los k para todos los @
    def get_precision_recall(self, group_recommendation_list, group, dataset, algorithm, aggregation_function, k=(10,11,10), threshold=.60, recall_depth=100):
        """
            Precision: Number of relevant elements recommended / total number of elements recommended.
                relevant elements recommended: Those whose real rating exceeds the threshold_value
            Precisión: Numero de elementos relevantes recomendados / numero total de elementos recomendados
                elementos relevantes recomendados: Aquellos cuyo rating real supera threshold_value

            Recall: Number of relevant recomended recomended elements / total number of relevant items
            Recall: Numero de elementos relevantes recomendados / Numero de elementos relevantes total
                Numero de elementos relevantes total: 
                                            V1: se asume que los k[2] elementos lo son porque si no no funcionaría para todas las agregaciones 
                                                # No sirve ya que 
                                            V2: Todos los elementos relevantes para un usuario en la lista top-N@k_max 
                                                #TODO: si se tiene los datos del usuario ordenados, elementos recomendados máximos
        """
        """
            {"k":number, "metric_name": "precision"|"recall", "metriic_value": number}
        """
        metrics = []
        try:
            threshold_value = threshold * dataset.rating_scale[1]
            
            real_relevant_items = 0
            predicted_relevant_items = 0
            true_positives = 0
            total_items = -1

            #Variables de depuración
            real_ratings = 0
            predicted_ratings = 0

            precision_2 = "err"
            recall_2 = "err"
            precision = "err"
            recall = "err"

            group_members_data = {}
            for user in group.users:
                group_members_data[user] = {
                    "true_positives": 0,
                    "predicted_relevant_items": 0,
                    "real_relevant_items": 0,
                    "all_relevant_items": self.get_user_all_relevant_items(user, threshold_value, dataset.df)
                }

            for group_item in group_recommendation_list:
                
                if total_items > k[1]:
                    break

                total_items += 1
                
                group_item_predicted_rating = group_recommendation_list[group_item]

                # # 1. Caso agregación primero, se almacenan todos los 
                users_real_ratings = []
                # Se obtienen los valores de los ratings de los usuarios para calcular el aggregado real
                for user in group.users:
                    # Seleccion valor real valoración de un item
                    real_rating = self.get_user_real_rating_from_df(user, group_item, dataset.df)
                    
                    users_real_ratings.append(real_rating)
                    #1. Metrica precisión V2 para cada usuario primero y luego media de precisiones
                    is_real_relevant = False
                    is_predicted_relevant = False

                    if real_rating > threshold_value:
                        is_real_relevant = True
                        group_members_data[user]["real_relevant_items"] += 1
                        
                    # Sin uso, solo funcionaría si agregación mantiene rating 0-4
                    # Se considerarían correctos si es predicción relevante y también para el usuario 
                    if group_item_predicted_rating > threshold_value:
                        is_predicted_relevant = True
                        group_members_data[user]["predicted_relevant_items"] += 1
                    if is_real_relevant and is_predicted_relevant:
                        group_members_data[user]["true_positives"] += 1

                # 2. Metrica precisión V1 agregación de valoraciones reales primero y luego comparación
                group_real_rating = aggregation_function(users_real_ratings)

                is_real_relevant = False
                is_predicted_relevant = False

                if group_real_rating > threshold_value:
                    is_real_relevant = True
                    real_relevant_items += 1

                # Sin uso, solo funcionaría si agregación mantiene rating 0-4
                # Se considerarían correctos si es predicción relevante y también para el usuario 
                if group_item_predicted_rating > threshold_value:
                    is_predicted_relevant = True
                    predicted_relevant_items += 1
                if is_real_relevant and is_predicted_relevant:
                    true_positives += 1



                # Cada vez que se alcanza un @k a representar, se imprime y se guarda la métrica
                if total_items in [actual_k for actual_k in range(k[0],k[1],k[2])]:
                    all_precisions = []
                    all_recalls = []
                    # 1. Metrica precisión V2 metrica media users, se calculan
                    for user in group.users:
                        precision_u, recall_u = self.calc_precision_recall(total_items, group_members_data[user]["real_relevant_items"], group_members_data[user]["all_relevant_items"] )
                        #precision_u, recall_u = self.calc_precision_recall(total_items, group_members_data[user]["true_positives"], group_members_data[user]["predicted_relevant_items"] )

                        all_precisions.append(precision_u)
                        all_recalls.append(recall_u)
                        # print("user", user, "k", total_items, precision_u)
                    
                    precision_2 = self.calc_list_average(all_precisions)
                    recall_2 = self.calc_list_average(all_recalls)

                    #2. metrica valoriones reales primero
                    precision, recall = self.calc_precision_recall(total_items, real_relevant_items, total_items)

                    # print("Precision and recall @"+str(total_items)+":", precision, recall)
                    # print("Precision and recall v2: @"+str(total_items)+":", precision_2, recall_2)
                    metrics.append(self.create_metric_dict(total_items, "precision", precision_2))
                    metrics.append(self.create_metric_dict(total_items, "recall", recall_2))
            

        except Exception as ex:
            print("get_precision_recall ex: ", ex)
            metrics.append(self.create_metric_dict(100, "precision", "err"))
            metrics.append(self.create_metric_dict(100, "recall", "err"))
        return metrics
    
    def calc_precision_recall(self, total_items, true_positives, relevant_items):
        """
            Given the amount of items in a top-n recomendation (total_items) the relevant items (true_positives) and
            all the relevant items for a certain user (relevant items), the precision and the recall are calculated and returned.

            total_items = tamaño @k
            true_positives = elmenetos relevantes para un usuario o grupo de ellos
            relevant items = elementos recomendados relevantes, se asume que son k_max para un usuario
        """
        if total_items == 0:
            #print("total_items: TREMENDO WARNING: DIVISION POR CERO PRECISION")
            total_items = -1

        precision = true_positives / total_items

        if (relevant_items == 0):
            #print("relevant_items: TREMENDO WARNING: DIVISION POR CERO PRECISION")
            relevant_items = 100000

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

    def get_mae_rsme(self, group_recommendation_list, group, dataset, algorithm, aggregation_function, threshold=0.6, k=(10,11,10)):
        """

            En recomendación para grupos también se puede considerar la diferencia entre items relevantes y predichos
        """
        metrics = []
        try:
            error_sum = 0
            error_squared_sum = 0

            total_items = -1

            #Variables de depuración

            group_members_data = {}
            for user in group.users:
                group_members_data[user] = {
                    "error_sum": 0,
                    "error_squared_sum": 0
                }

            for group_item in group_recommendation_list:
                if total_items > k[1]:
                    break
                total_items += 1

                group_item_predicted_rating = group_recommendation_list[group_item]

                users_real_ratings = []
                # Se obtienen los valores de los ratings de los usuarios para calcular el aggregado real
                for user in group.users:
                    # Seleccion valor real valoración de un item
                    real_rating = self.get_user_real_rating_from_df(user, group_item, dataset.df)

                    # Si para un usuario no existiese una valoración real se ignora
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

                # Cada vez que se alcanza un @k a representar, se imprime y se guarda la métrica
                if total_items in [actual_k for actual_k in range(k[0],k[1],k[2])]:
                    #TODO comprobación errores
                    mae_2 = group_members_data[user]["error_sum"] /total_items
                    rmse_2 = sqrt(group_members_data[user]["error_squared_sum"] /total_items)

                    mae = error_sum/total_items
                    rmse = sqrt(error_squared_sum/total_items)

                    print("MAE y RMSE: @"+str(total_items)+":", mae, rmse)
                    print("MAE y RMSE v2: @"+str(total_items)+":", mae_2, rmse_2)

                    metrics.append(self.create_metric_dict(total_items, "mae", mae))
                    metrics.append(self.create_metric_dict(total_items, "rmse", rmse))
                    metrics.append(self.create_metric_dict(total_items, "mae_2", mae_2))
                    metrics.append(self.create_metric_dict(total_items, "rmse_2", rmse_2))

        except Exception as ex:
            print("get_mae_rsme ex: ", ex)
            metrics.append(self.create_metric_dict(total_items, "mae", "err"))
            metrics.append(self.create_metric_dict(total_items, "rmse", "err"))
            metrics.append(self.create_metric_dict(total_items, "mae_2", "err"))
            metrics.append(self.create_metric_dict(total_items, "rmse_2", "err"))
        return metrics

    #TODO: Se pueden dejar todos los mismos valores en todas las métricas para homogeneizarlo o quitar las necesarias
    def get_dcg_ndcg(self, group_recommendation_list, group, dataset, algorithm, aggregation_function, threshold=0.6, k=10):
        metrics = []
        try:
            threshold_value = threshold * dataset.rating_scale[1]

            total_items = -1
            dcg = 0
            idcg = 0

            group_members_data = {}
            for user in group.users:
                group_members_data[user] = {
                    "relevance_list": [],
                    "dcg": 0,
                    "idcg": 0,
                    "ndcg": 0,
                }

            # 1. Para cada usuario se obtiene una lista de relevancias [0,1,1,1,0,0,1,1,1,0]
            for group_item in group_recommendation_list:
                if total_items > k[1]:
                    break
                total_items += 1 #Aqui también es posición en la lista (empezando en 0)

                for user in group.users:
                    real_rating = self.get_user_real_rating_from_df(user, group_item, dataset.df)

                    if real_rating >= threshold_value:
                        group_members_data[user]['relevance_list'].append(1)
                    else:
                        group_members_data[user]['relevance_list'].append(0)

            # por cada @k a representar, se imprime y se guarda la métrica
            for actual_k in range(k[0],k[1],k[2]):
                # 2. Una vez se tiene la lista de relevancias, se puede calcular el dcg
                avg_dcg = 0
                avg_ndcg = 0
                for user in group.users:
                    for i in range(actual_k):
                        group_members_data[user]['dcg'] += (2**(group_members_data[user]['relevance_list'][i])-1) / log(1+i+1, 2)
                        group_members_data[user]['idcg'] += 1 / log(1+1+1, 2)
                    group_members_data[user]['ndcg'] = group_members_data[user]['dcg'] / group_members_data[user]['idcg']

                    avg_dcg += group_members_data[user]['dcg']
                    avg_ndcg += group_members_data[user]['ndcg']
                avg_dcg = avg_dcg / len(group.users)
                avg_ndcg = avg_ndcg / len(group.users)
                # print("DCG y nDCG @"+str(actual_k)+":", avg_dcg, avg_ndcg)

                metrics.append(self.create_metric_dict(actual_k, "dcg", avg_dcg))
                metrics.append(self.create_metric_dict(actual_k, "ndcg", avg_ndcg))


        except Exception as ex:
            print("get_dcg_ndcg ex: ", ex)
            metrics.append(self.create_metric_dict(100, "dcg", "err"))
            metrics.append(self.create_metric_dict(100, "ndcg", "err"))
        return metrics


    def get_coverage(self, group_recommendation_list, group, dataset, algorithm, aggregation_function, k=10, threshold=.60):
        '''
            #TODO
        '''
        try:
            return 0
        except Exception as ex:
            print("get_coverage ex: ", ex)
            return "err", "err"

    def get_serendipity(self, group_recommendation_list, group, dataset, algorithm, aggregation_function, k=10, threshold=.60):
        '''
            #TODO
        '''
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
