import pandas as pd
from surprise import Dataset as Surprise_dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold

"""
    Class in charge of loading and processing the dataset
        Input: Path of the dataset containing the tuple user-item-rating (optionally context information)
               + other configurations
        Output: Pandas dataframe with the user-item-rating array
               + data variable with formats required by the libraries
               + train/test split if needed 

    Modulo encargado de la gestión de datasets
        Input: Archivo donde se encuentra el dataset con los user-item-rating
               + Otras configuraciones   
        Output: Pandas dataframe con la matriz user-item-rating
               + variable data con formatos requeridos por las librerías
               + train/test si se necesitan 
"""

class Dataset():
    def __init__(
        self, filepath, format="csv", sep=";", rating_scale=(0,4), 
        cols=None, del_cols=None, lib="surprise", line_format='user item rating',
        prefilter_columns=[],
        split_dataset=False, test_size=.25,
        verbose=True, full_verbose=False
    ):
        """
        Constructor
            :param filepath: complete path to where the dataset is located (absolute or relative)
            :type string:
            
            :param format: file format
            :type string:

            :param separator: dataset separator type
            :type char:

            :param rating_scale: minimum and maximum ratings
            :type tuple(min, max):

            :param line_format: Format of the fields in the user-item-rating columns, some libraries allow to choose the order
            :type string with format 'user item rating'

            :param cols: Columns used for recommendation
            :type string list:

            :param del_cols: Unused columns to be deleted, if you prefer to select those to be deleted instead of cols
            :type string list:

            :param prefilter_columns: Pre-filtered columns to be removed from the dataset
            :type filter_tuple list: i.e: [{"column": 'acousticness', "mode": "value", "threshold": 0.8, "direction": "below"}]

            :param lib: Library used
            :type string: "surprise"

            :param split_dataset: train/test split  P.ej. Surprise
            :type boolean:

            :param test_size: train/test ratio
            :type 0-1:
        """
        self.module = "Dataset: "

        if not filepath or filepath == None:
            print("A dataset must be entered")
            return

        # Pandas dataframe with the data loaded
        self.df = None
        # Data processed for a library. E.g. surprise
        self.data = None
        self.prefilter_columns = prefilter_columns
        # Training and test sets
        self.split_dataset = split_dataset
        self.test_size = test_size
        self.trainset = None
        self.testset = None
        self.kfolds = None

        self.lib = lib

        self.verbose = verbose
        self.full_verbose = full_verbose

        self.filepath = filepath
        self.format = format
        self.sep = sep

        self.rating_scale = rating_scale
        self.line_format = line_format
        self.cols = cols
        self.del_cols = del_cols

        # 1. Se carga el dataset a partir del fichero en self.df
        self.read_dataset()
        
        # 2.1 Se realiza el prefiltrado de self.df
        self.filter_dataset()

        # 2.2 Se eliminan las columnas innecesarias de self.df
        self.clear_columns()

        # 3. Se carga en el modelo, se hace split de datos, etcétera.
        self.load_dataset()
        self.create_train_test_split()
        self.create_k_fold()

        # 4.x. Si se desea, se imprime el dataset cargado
        if (self.full_verbose and self.verbose) : self.printDataset()

    def read_dataset(self):
        """
            Carga en self.df el dataset en formato pandas dataframe. 
            Available formats: .CSV
        """
        if self.format == "csv":
            if self.verbose : print(self.module + "Reading dataset " + self.filepath + " with sep " + self.sep)
            self.df = pd.read_csv(self.filepath, sep=self.sep)
            if self.verbose : print(self.module + "Dataset readed: ", self.df.shape)
        else:
            if self.verbose : print(self.module + "Invalid format.")

    def filter_dataset(self):
        """
            Filter the dataset using tuples in the following format:
                {"column": None, "mode": "value", "threshold": 1, "direction": "below"}
            where the options are: 
                column: name of the column to filter
                mode: 'value' or 'nominal' 
                    - depending on whether filtering is based on numerical or nominal values, #TODO: nominal filtering
                threshold: number from [0-1]
                direction: "below" or "above" 
                    - in a numerical filtering, to take the value lower or higher than the threshold value

            :param filter_columns: Array of columns to filter
            :type filter_tuple list:
        """
        
        for filter in self.prefilter_columns:
            try:
                if filter['column'] == None:
                    pass
                else:
                    if filter['mode'] == "value":
                        print("Filtering:", filter)
                        if filter['direction'] == "above":
                            self.df.drop(self.df.loc[self.df[filter['column']]<filter['threshold']].index, inplace=True)
                        if filter['direction'] == "below":
                            self.df.drop(self.df.loc[self.df[filter['column']]>filter['threshold']].index, inplace=True)
                        if self.verbose : print(self.module + "Dataset filtered: ", self.df.shape)
                    if filter['mode'] == "nominal":
                        if self.verbose : print(self.module + "Nominal filtering not implemented.")
                        pass

            except Exception as ex:
                print("FILTERING ERROR: ", filter, ex)

        pass

    def clear_columns(self):
        """
            Clears the unnecessary columns from the loaded dataset.
            Both necessary and unnecessary columns can be passed.
            The necessary ones will be considered first, if they have not been entered, we will try to eliminate the unnecessary ones.

            :param cols: array of necessary columns (strings)
            :type string list:

            :param del_cols: array of unnecessary columns (strings)
            :type string list:

            #TODO: nominal filtering
        """
        if self.cols == None and self.del_cols == None:
            if self.verbose : print(self.module + "No columns have been set to be discarded, 3 are assumed by default.")
            # TODO: Comprobación 3 columnas
            if len(list(self.df.columns)) != 3:
                print(self.module + "Error: there are not only 3 columns: user-item-rating")
            return -1

        elif self.cols != None:
            self.del_cols = list(self.df.columns)
            for col in self.cols: 
                if col in self.del_cols : self.del_cols.remove(col)

        for del_column in self.del_cols:
            try:
                del self.df[del_column]
            except Exception as exception:
                print(self.module + "Error deleting the column " + del_column)
                print(exception)

    def load_dataset(self):
        """
            Loads the dataset in order to be used for a specific library

            :param lib: library
            :type string: 

            :param various_args: various args depending on the library used e.g. line_format, rating_scale... 
            :type various:

            #TODO: implementation of new libraries
        """
        if self.lib == "surprise":
            if self.verbose : print(self.module + "Loading data for " + self.lib)
            reader = Reader(line_format=self.line_format, sep=self.sep, rating_scale=self.rating_scale)
            self.data = Surprise_dataset.load_from_df(self.df, reader=reader)
            if self.verbose : print(self.module + "Loaded")
        else:
            if self.verbose : print(self.module + "load_dataset() without using any library")

    def create_train_test_split(self):
        """
            stores in self.trainset and self.testset a pair of test/train set

            :param lib: the library used may be require a specific format
            :type string: 

            :param testsize: testsize
            :type number: 

            :param random_state: to generate replicable experiments, the random state has been set to 0
            :type number:
        """
        if self.lib == "surprise" and self.split_dataset:
            if self.verbose : print(self.module + "Generating train/test split for " + self.lib + " with rate: " + str(self.test_size))
            self.trainset, self.testset = train_test_split(self.data, test_size=self.test_size, random_state=0)
            # Other ways considered in the surprise docs:
            # fulltrainset = data.build_full_trainset()
            # antitestset = fulltrainset.build_anti_testset()
            if self.verbose : print(self.module + "splited")
        else:
            if self.verbose : print(self.module + "No libraries are used and no split_dataset generated")


    def create_k_fold(self, k=4):
        """
            #TODO: due to the large number of possibilities in the recommendation for groups using different aggregation methods, 
            algorithms, top-n list sizes, context types... no tests using cross validation have been performed.
            however, its implementation should be reconsidered depending on the experiment to be carried out.

            Prints the pandas dataframe, optionally a prefix_text or/and sufix_text can be passed.
        """
        if self.lib == "surprise" and self.split_dataset:
            # Utilizar cross validation
            if self.verbose : print(self.module + "Generating crossvalidation split for " + self.lib + " with splits: " + str(k))
            # https://surprise.readthedocs.io/en/stable/getting_started.html#use-cross-validation-iterators
            kf = KFold(n_splits=k, random_state=0)
            self.kfolds = kf.split(self.data)
            if self.verbose : print(self.module + "kfold splited in " + str(k))
            # for t, i in self.kfolds:
            #     print(t)
            #     print(i)
        else:
            if self.verbose : print(self.module + "K fold not implemented for " + self.lib )


    def printDataset(self, prefix_text=False, sufix_text=False):
        """
            Prints the pandas dataframe, optionally a prefix_text or/and sufix_text can be passed.
        """
        if prefix_text : print(prefix_text)
        print(self.df)
        if sufix_text : print(sufix_text)
