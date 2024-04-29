import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, log_loss
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import warnings
warnings.filterwarnings("ignore")

class ImportData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self, delimiter=','):
        self.data = pd.read_csv(self.file_path, delimiter=delimiter)

    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

class DataPreprocessing:
    def __init__(self, input_df, output_df):
        self.input_df = input_df
        self.output_df = output_df
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def drop_columns(self, column_names):
        for column_name in column_names:
            if column_name in self.input_df.columns:
                self.input_df.drop(columns=[column_name], inplace=True)

    def split_dataset(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.input_df, self.output_df, test_size=test_size, random_state=random_state)

    def fill_missing_values(self, column_name, method='mean'):
        if method == 'mean':
            fill_value = self.x_train[column_name].mean()
        elif method == 'median':
            fill_value = self.x_train[column_name].median()
        elif method == 'mode':
            fill_value = self.x_train[column_name].mode()[0]
        else:
            return None
        self.x_train[column_name].fillna(fill_value, inplace=True)
        self.x_test[column_name].fillna(fill_value, inplace=True)

    def encode_feature(self, column_name):
        self.x_train = pd.get_dummies(self.x_train, columns=[column_name])
        self.x_test = pd.get_dummies(self.x_test, columns=[column_name])

    def replace_categorical(self, train_encode, test_encode):
        self.x_train.replace(train_encode, inplace=True)
        self.x_test.replace(test_encode, inplace=True)

    def scale_data(self, scaler, columns):
        self.x_train[columns] = scaler.fit_transform(self.x_train[columns])
        self.x_test[columns] = scaler.transform(self.x_test[columns])

class Modelling:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None

    def train_XGB(self, objective='binary:logistic', learning_rate= 0.1, max_depth= 3, n_estimators= 200, random_state=42):
        self.model = xgb.XGBClassifier(objective=objective, learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators, random_state = random_state)
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        if self.model is None:
            print("Model Untrained")
            return
        y_predict = self.model.predict(self.x_test)
        print('\nClassification Report\n')
        print(classification_report(self.y_test, y_predict))

    def save_model(self, filename):
      with open(filename, 'wb') as file:
        pickle.dump(self.model, file)

"""**Usage**"""

#Import New Dataset
data_importer = ImportData('data_C.csv')
data_importer.load_data()
data_importer.create_input_output('churn')

#Define The Input and Output
input_data = data_importer.input_df
output_data = data_importer.output_df

# Preprocess The Data
preprocessor = DataPreprocessing(input_data, output_data)

#Drop Irrelevant Data
preprocessor.drop_columns(['Unnamed: 0', 'id', 'CustomerId', 'Surname'])

# Dataset Splitting
preprocessor.split_dataset()
x_train = preprocessor.x_train
x_test = preprocessor.x_test
y_train = preprocessor.y_train
y_test = preprocessor.y_test

# Replace categorical to Numerical
train_encode = {"Gender": {"Male": 1, "Female": 0}}
test_encode = {"Gender": {"Male": 1, "Female": 0}}
preprocessor.replace_categorical(train_encode, test_encode)

#One Hot Encoding
preprocessor.encode_feature('Geography')

# impute Missing Value
preprocessor.fill_missing_values(column_name = 'CreditScore', method = 'mean')

# Scaling Dataset
preprocessor.scale_data(RobustScaler(), ['Age', 'CreditScore'])
preprocessor.scale_data(MinMaxScaler(), ['Balance', 'EstimatedSalary'])

# Create an instance of Modelling
model = Modelling(preprocessor.x_train, preprocessor.y_train, preprocessor.x_test, preprocessor.y_test)

# Train Decision Tree Classifier with default parameters
model.train_XGB()

# Evaluate the trained model
model.evaluate_model()

#model.save_model('BestModel_XGB.pkl')
