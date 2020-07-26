import numpy as np
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_PassengerId = test_df['PassengerId']
combine_df = [train_df, test_df]

for dataset in combine_df:
    # Name / Title
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Mme', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace(['Don', 'Dona', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', \
                                                 'Col', 'Capt', 'Countess', 'Jonkheer'], 'Other')
    dataset['Title'] = dataset['Title'].map({'Mr': 1, 'Mrs': 2, 'Master': 3, 'Miss': 4, 'Other': 5})
    dataset['Title_1'] = dataset['Title'] == 1
    dataset['Title_2'] = dataset['Title'] == 2
    dataset['Title_3'] = dataset['Title'] == 3
    dataset['Title_4'] = dataset['Title'] == 4
    dataset['Title_5'] = dataset['Title'] == 5

    # Convert categorical Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1})

    # Impute Age based on Pclass and Sex
    guess_age = dataset.dropna().groupby(['Pclass', 'Sex'])['Age'].median()
    for s in range(0, 2):
        for pc in range(1, 4):
            dataset.loc[(dataset.Sex == s) & (dataset.Pclass == pc) & (dataset.Age.isnull()), \
                        'Age'] = guess_age[pc][s]
    dataset['Age'] = dataset['Age'].astype(int)
    # Age banding
    dataset['Agebins'] = pd.qcut(dataset['Age'], 5, labels=[0, 1, 2, 3, 4]).astype(int)
    dataset['Agebins_0'] = dataset['Agebins'] == 0
    dataset['Agebins_1'] = dataset['Agebins'] == 1
    dataset['Agebins_2'] = dataset['Agebins'] == 2
    dataset['Agebins_3'] = dataset['Agebins'] == 3

    dataset['Pclass_1'] = dataset['Pclass'] == 1
    dataset['Pclass_2'] = dataset['Pclass'] == 2

    # Embarked
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset['Embarked_S'] = dataset.Embarked == 'S'
    dataset['Embarked_C'] = dataset.Embarked == 'C'

    # Fare
    guess_fare = dataset.dropna().groupby(['Pclass'])['Fare'].median()
    for pc in range(1, 4):
        dataset.loc[(dataset.Pclass == pc) & (dataset.Fare.isnull()), 'Fare'] = guess_fare[pc]
    dataset['Farebins'] = pd.qcut(dataset['Fare'], 5, labels=[0, 1, 2, 3, 4]).astype(int)
    dataset['Farebins_0'] = dataset['Farebins'] == 0
    dataset['Farebins_1'] = dataset['Farebins'] == 1
    dataset['Farebins_2'] = dataset['Farebins'] == 2
    dataset['Farebins_3'] = dataset['Farebins'] == 3

    # Family stuff
    dataset['Familysize'] = dataset['Parch'] + dataset['SibSp']
    dataset['isAlone'] = dataset['Familysize'] == 1

# Done with dataset preparation
corr = train_df.corr()['Survived'].abs().sort_values(ascending=False)
#print(corr)

# ML Modelling starts here ->
# Boosting ensemble methods


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

# Define our important feature columns
feature_columns = ['Title_1', 'Title_2', 'Title_3', 'Title_4', 'Pclass_1', 'Pclass_2', 'isAlone', 'Sex', \
                   'Agebins_0', 'Agebins_1', 'Agebins_2', 'Agebins_3', \
                   'Farebins_0', 'Farebins_1', 'Farebins_2', 'Farebins_3', \
                   'Embarked_S', 'Embarked_C']

X = train_df[feature_columns]
y = train_df['Survived']
# Using stratify=y here because Survived is 60/40% approx split, so we want it represented
# in both train & test datasets correctly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)



dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(1,20,1)
}
adaboost_param_grid = {
    'n_estimators': np.arange(50, 300, 50),
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]
}
gb_param_grid = {
    'n_estimators': np.arange(50, 300, 50),
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'max_depth': np.arange(1,10,2),
    'subsample': np.arange(1.0, 0.6, -0.1)
}


dtgrid = GridSearchCV(DecisionTreeClassifier(),
                      param_grid=dt_param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
dtgrid.fit(X_train, y_train)
print('DecisionTree Tuned')
print('Best DT Params: ', dtgrid.best_params_)
print('Best DT Score (ROC AUC): ', dtgrid.best_score_)

adagrid = GridSearchCV(AdaBoostClassifier(base_estimator=dtgrid.best_estimator_),
                       param_grid=adaboost_param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
adagrid.fit(X_train, y_train)
print('Adaboost Tuned')
print('Best Parameters: ', adagrid.best_params_)
print('Best Score: (ROC AUC): ', adagrid.best_score_)

y_pred = adagrid.predict(X_test)
y_pred_proba = adagrid.predict_proba(X_test)[:,1]
print('Test ROC AUC: ', roc_auc_score(y_test,y_pred_proba))
print('Test F1: ', f1_score(y_test,y_pred))
print('Test Accuracy: ', accuracy_score(y_test,y_pred))

y_pred_final = adagrid.predict(test_df[feature_columns])
submission = pd.DataFrame({'PassengerId': test_PassengerId, 'Survived': y_pred_final})
filename = 'submission 13 - Tuned Adaboost with Tuned DT.csv'
submission.to_csv(filename, index=False)
print('Created submission file: ', filename)





gbgrid = GridSearchCV(GradientBoostingClassifier(),
                      param_grid=gb_param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
gbgrid.fit(X_train, y_train)
print('GradientBoost Tuned')
print('Best Parameters: ', gbgrid.best_params_)
print('Best Score: (ROC AUC): ', gbgrid.best_score_)

y_pred = gbgrid.predict(X_test)
y_pred_proba = gbgrid.predict_proba(X_test)[:,1]
print('Test ROC AUC: ', roc_auc_score(y_test,y_pred_proba))
print('Test F1: ', f1_score(y_test,y_pred))
print('Test Accuracy: ', accuracy_score(y_test,y_pred))

y_pred_final = gbgrid.predict(test_df[feature_columns])
submission = pd.DataFrame({'PassengerId': test_PassengerId, 'Survived': y_pred_final})
filename = 'submission 13 - Tuned Gradient Boost.csv'
submission.to_csv(filename, index=False)
print('Created submission file: ', filename)
