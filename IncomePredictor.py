from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from EncodeCategorical import EncodeCategorical
from ImputeCategorical import ImputeCategorical
from PredictorUtils import PredictorUtils
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from elm import ELMClassifier, ELMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns



filename='/Users/ridvansalih/Desktop/ADULT_TRAIN_DATA_TABLE.csv'


dataset = PredictorUtils.load_data(filename)
PredictorUtils.visualize_data(dataset.rawdata)

# Encode our target data
yencode = LabelEncoder().fit(dataset.train_target)

# Construct the preprocessing pipeline
pre = Pipeline([
            ('encoder',  EncodeCategorical(dataset.categorical_features.keys())),
            ('imputer', ImputeCategorical(['workclass', 'native-country', 'occupation'])),
            ('scaler', StandardScaler())
        ])


# Fit the preprocessing on pipeline
pre.fit(dataset.data,yencode.transform(dataset.data_target))

# prepare training models
models = []

models.append(('RFC', RandomForestClassifier()))
models.append(('ELM', ELMClassifier(regressor=ELMRegressor())))
models.append(('MLP', MLPClassifier()))
models.append(('LRC', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NBC', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
bestmodel=None
bestscore=0
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, pre.transform(dataset.train),yencode.transform(dataset.train_target), cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)

    if (sum(cv_results) > bestscore):
        bestscore = sum(cv_results)
        bestmodel = model
    print("MODEL: {}, ACCURACY: {:.4f} (+/-{:.4f})".format(name, cv_results.mean(), cv_results.std() / 2))

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



# Fit the model
#parameters = [{'n_estimators': [10,20,30,40,50]},]

#clf = GridSearchCV(RandomForestClassifier(n_estimators=1), parameters, cv=5, scoring='accuracy')
#clf.fit(pre.transform(dataset.train),yencode.transform(dataset.train_target))
#for params, mean_score, scores in clf.grid_scores_:
#    print("{}: {:.3f} (+/-{:.3f})".format(params, mean_score, scores.std() / 2))
#print("The best model for RF found has n_estimator={}, SCORE={}".format(clf.best_estimator_.n_estimator,clf.best_score_))


# Test the model
bestmodel.fit(pre.transform(dataset.train),yencode.transform(dataset.train_target))

# Encode test targets, and strip trailing '.'
y_true = yencode.transform([y.rstrip(".") for y in dataset.target_test])

    # Use the model to get the predicted value
y_pred = bestmodel.predict(pre.transform(dataset.test))

    # execute classification report
    # print classification_report(y_true, y_pred, target_names=dataset.target_names)
cr = classification_report(y_true, y_pred, target_names=dataset.target_names)
print(cr)
PredictorUtils.plot_classification_report(cr)
PredictorUtils.plot_feature_relations(bestmodel,dataset.train)

