# BRIEF INFO ABOUT THE PROJECT
The adult data income prediction project is developed on Python by using PyCharm IDE. The project has 6 files:

1.	IncomePredictor: The main script of the project. It loads data from CSV files, and parses into a usable format for machine learning.  Categorical fields in the given data is encoded to numerical format followed by imputation of missing values. Finally scaling of data is realized in the pre-processing step. 
As described also in the script, 9 different supervised classification methods are compared by using 10-fold cross validation technique and default hyper-parameters of these methods. Although fine-tuning via grid-search on the parameters is planned and implemented in the file, it is not used due to limited time for this effort. If the tester of the file want to check the result of the grid-search, he/she can just remove comment-out signs on the code block. 
At the end, the file executes the prediction on the test data by using the best learning model selected after cross validation, and plots performance graphs and correlations among the features. 
2.	PredictorUtils: It is a utility class including data visualisation, classification reporting and data loading functions.
3.	EncodeCategorical: The class selects the categorical features and encode them in order to use with the rest of numerical features in the dataset while modelling the predictor.
4.	ImputeCategorical: The purpose of the class is to fill missing values on some columns.
5.	elm: Third party ML tool which is not included in built-in libraries of scikit-learn
6.	random_layer: Third party functions used by Extreme Learning Machine in elm

The project can be tested by using the data samples on: http://archive.ics.uci.edu/ml/datasets/Census+Income
