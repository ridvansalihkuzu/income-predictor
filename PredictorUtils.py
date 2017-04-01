import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.datasets.base import Bunch
import sklearn.cross_validation as cross_validation
from sklearn.preprocessing import LabelEncoder


class PredictorUtils:

    @staticmethod
    def plot_classification_report(cr, title=None, cmap=cm.YlOrRd):
        title = title or 'Classification report'
        lines = cr.split('\n')
        classes = []
        matrix = []

        for line in lines[2:(len(lines) - 3)]:
            s = line.split()
            classes.append(s[0])
            value = [float(x) for x in s[1: len(s) - 1]]
            matrix.append(value)

        fig, ax = plt.subplots(1)

        for column in range(len(matrix) + 1):
            for row in range(len(classes)):
                txt = matrix[row][column]
                ax.text(column, row, matrix[row][column], va='center', ha='center')

        fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        x_tick_marks = np.arange(len(classes) + 1)
        y_tick_marks = np.arange(len(classes))
        plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
        plt.yticks(y_tick_marks, classes)
        plt.ylabel('Classes')
        plt.xlabel('Measures')
        plt.show()

    @staticmethod
    def plot_feature_relations(classifier,train_data):
        coefs = pd.Series(classifier.coef_[0], index=train_data.columns)
        coefs.sort()
        coefs.plot(kind="bar")
        plt.show()

    @staticmethod
    def load_data(filename):

        names = [
            'ID',
            'age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country',
            'income',
            'target',
        ]
        arr=list(range(1, 16))
        data = pd.read_csv(filename, sep="\s*,", engine='python',names=names,skiprows=[0],usecols=arr)
        print(data.describe())

        meta = {
            'target_names': list(data.income.unique()),
            'feature_names': list(data.columns),
            'categorical_features': {
                column: list(data[column].unique())
                for column in data.columns
                if data[column].dtype == 'object'
            },
        }

        names = meta['feature_names']
        meta['categorical_features'].pop('income')

        train, test = cross_validation.train_test_split(data, test_size=0.25)

        # Return the bunch with the appropriate data chunked apart
        return Bunch(
            rawdata=data,
            data=data[names[:-1]],
            data_target=data[names[-1]],
            train=train[names[:-1]],
            train_target=train[names[-1]],
            test=test[names[:-1]],
            target_test=test[names[-1]],
            target_names=meta['target_names'],
            feature_names=meta['feature_names'],
            categorical_features=meta['categorical_features'],
        )

    @staticmethod
    def visualize_data(data):

        encoded_data, _ = PredictorUtils.number_encode_features(data)
        sns.heatmap(encoded_data.corr(), square=True)
        plt.show()

        sns.countplot(y='occupation', hue='income', data=data, )
        sns.plt.title('Occupation vs Income')
        sns.plt.show()

        sns.countplot(y='education', hue='income', data=data, )
        sns.plt.title('Education vs Income')
        sns.plt.show()

        # How years of education correlate to income, disaggregated by race.
        # More education does not result in the same gains in income
        # for Asian Americans/Pacific Islanders and Native Americans compared to Caucasians.
        g = sns.FacetGrid(data, col='race', size=4, aspect=.5)
        g = g.map(sns.boxplot, 'income', 'education-num')
        #sns.plt.title('Years of Education vs Income, disaggregated by race')
        sns.plt.show()

        # How years of education correlate to income, disaggregated by sex.
        # More education also does not result in the same gains in income for women compared to men.
        g = sns.FacetGrid(data, col='sex', size=4, aspect=.5)
        g = g.map(sns.boxplot, 'income', 'education-num')
        #sns.plt.title('Years of Education vs Income, disaggregated by sex')
        sns.plt.show()

        # How age correlates to income, disaggregated by race.
        # Generally older people make more, except for Asian Americans/Pacific Islanders.
        g = sns.FacetGrid(data, col='race', size=4, aspect=.5)
        g = g.map(sns.boxplot, 'income', 'age')
        #sns.plt.title('Age vs Income, disaggregated by race')
        sns.plt.show()

        # How hours worked per week correlates to income, disaggregated by marital status.
        g = sns.FacetGrid(data, col='marital-status', size=4, aspect=.5)
        g = g.map(sns.boxplot, 'income', 'hours-per-week')
        #sns.plt.title('Hours by week vs Income, disaggregated by marital status')
        sns.plt.show()

        sns.violinplot(x='sex', y='education-num', hue='income', data=data, split=True, scale='count')
        sns.plt.title('Years of Education and Sex vs Income')
        sns.plt.show()

        sns.violinplot(x='sex', y='hours-per-week', hue='income', data=data, split=True, scale='count')
        sns.plt.title('Hours-per-week and Sex vs Income')
        sns.plt.show()

        sns.violinplot(x='sex', y='age', hue='income', data=data, split=True, scale='count')
        sns.plt.title('Age and Sex vs Income')
        sns.plt.show()

        g = sns.PairGrid(data,
                         x_vars=['income', 'sex'],
                         y_vars=['age'],
                         aspect=.75, size=3.5)
        g.map(sns.violinplot, palette='pastel')
        sns.plt.show()

        g = sns.PairGrid(data,
                         x_vars=['marital-status', 'race'],
                         y_vars=['education-num'],
                         aspect=.75, size=3.5)
        g.map(sns.violinplot, palette='pastel')
        sns.plt.show()

    @staticmethod
    def number_encode_features(df):
        result = df.copy()
        encoders = {}
        for column in result.columns:
            if result.dtypes[column] == np.object:
                encoders[column] = LabelEncoder()
                result[column] = encoders[column].fit_transform(result[column])
        return result, encoders


