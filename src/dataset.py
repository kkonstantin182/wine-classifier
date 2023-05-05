import pandas as pd
from constants import SEED, TRAIN_SIZE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Dataset:

    def __init__(self, dataset_path, target_map: dict, columns_names: dict):
        """
        Initializes the Dataset object.

        Parameters:
        dataset_path (str): the path to the dataset file
        target_map (dict): a dictionary that maps target labels to integers
        columns_names (dict): a dictionary that contains lists of column names for different types of features

        """
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path)
        self.dataset.columns = self.dataset.columns.str.lower()

        self.train_set = list()
        self.test_set = list()

        self.target_map = target_map

        self.target_name = columns_names.get('target', [])
        self.numerical_names = columns_names.get('numerical', [])
        self.categorical_names = columns_names.get('categorical', [])
        self.text_names = columns_names.get('text', [])

        columns_to_keep = [col for col in [*self.target_name, *self.text_names, *self.numerical_names, *self.categorical_names] if col is not None]

        self.dataset = self.dataset[columns_to_keep]

        print(self.print_shape)

    def __call__(self):
        """
        Splits the dataset into training and testing sets, encodes the target labels, and returns the resulting datasets.

        Returns:
        (tuple): a tuple of training and testing datasets
        """
        self._train_test_sets_split()
        self._encode_target()
        print(self.print_shape())
        return self.train_set, self.test_set
         

    def print_shape(self):
        """
        Prints the shape of the training and testing sets.
        """
        return f"The train shape: {self.train_set.shape}\nThe test shape: {self.test_set.shape}"
        
    def _train_test_sets_split(self, train_size=TRAIN_SIZE, seed=SEED):
        """
        Splits the dataset into training and testing sets.

        Parameters:
        train_size (float): the size of the training set
        seed (int): random seed for the split

        Returns:
        (tuple): a tuple of training and testing datasets
        """
        self.train_set, self.test_set = train_test_split(self.dataset, 
                                                         test_size= 1-train_size, 
                                                         random_state=seed,
                                                         stratify=self.dataset[self.target_name])
        return self.train_set, self.train_set
    

    def _encode_target(self):
        """
        Encodes the target labels using the target map.
        """
        self.train_set[self.target_name[0]] = self.train_set[self.target_name[0]].map(self.target_map)
        self.test_set[self.target_name[0]] = self.test_set[self.target_name[0]].map(self.target_map)
        return self.train_set, self.test_set

    def _encode_cat_var(self):
        """
        Encodes categorical variables using one-hot encoding.
        """
        pass
