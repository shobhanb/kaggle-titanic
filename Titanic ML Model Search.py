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


# ML Modelling starts here ->
# Let's try a basic fit for ALL classification models for comparison purposes
# Then slowly start tuning them


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

# Define our important feature columns
feature_columns = ['Title_1', 'Title_2', 'Title_3', 'Title_4', 'Pclass_1', 'Pclass_2', 'isAlone', 'Sex', \
                   'Agebins_0', 'Agebins_1', 'Agebins_2', 'Agebins_3', \
                   'Farebins_0', 'Farebins_1', 'Farebins_2', 'Farebins_3', \
                   'Embarked_S', 'Embarked_C']

X = train_df[feature_columns]
y = train_df['Survived']
# Using stratify=y here because Survived is 60/40% approx split, so we want it represented
# in both train & test datasets correctly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


logreg_param_grid = {
    'penalty': ['l1', 'l2'],
    'C': np.arange(0.1, 1, 0.2),
    'max_iter': np.arange(20,100,10),
    'solver': ['liblinear']
}
knn_param_grid = {
    'n_neighbors': np.arange(2,10,1),
    'leaf_size': np.arange(5,50,5),
    'p': np.arange(1.0, 2.0, 0.25)
}
nb_param_grid = {}
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(1,20,1)
}
svc_param_grid = {
    'C': np.arange(0.1, 1, 0.1),
    'kernel': ['linear', 'poly', 'rbf','sigmoid']
}
rf_param_grid = {
    'n_estimators': np.arange(10,100,10),
    'criterion': ['entropy'],
    'max_depth': np.arange(1, 20, 1)
}

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(),
    GaussianNB(),
    DecisionTreeClassifier(),
    SVC(probability=True),
    RandomForestClassifier(),
]

clf_names = [
    'LogisticRegression',
    'KNeighborsClassifier',
    'GaussianNB',
    'DecisionTreeClassifier',
    'SVC',
    'RandomForestClassifier',
]

clf_params = [logreg_param_grid, knn_param_grid, nb_param_grid, dt_param_grid, svc_param_grid, rf_param_grid ]

scores = pd.DataFrame(index=clf_names, columns=['Accuracy', 'F1', 'ROC AUC'])


for name, clf, params in zip(clf_names, classifiers, clf_params):
    clf_cv = GridSearchCV(clf, param_grid=params, scoring='f1', cv=5, n_jobs=-1, verbose=0)
    clf_cv.fit(X_train, y_train)
    print('Fitting :', name)
    print('Best Parameters: ', clf_cv.best_params_)
    print('Best Score: (F1)', clf_cv.best_score_)

    y_pred = clf_cv.predict(X_test)
    y_pred_proba = clf_cv.predict_proba(X_test)[:,1]
    scores.loc[name,'ROC AUC'] = roc_auc_score(y_test, y_pred)
    scores.loc[name,'F1'] = f1_score(y_test, y_pred)
    scores.loc[name, 'Accuracy'] = accuracy_score(y_test, y_pred)
    scores.loc[name, 'Model'] = clf_cv.best_estimator_


scores.sort_values(by='ROC AUC', ascending=False, inplace=True)
print(scores.iloc[:,:-1])

# Pick DT as the classifier for the job

best_clf = scores.iloc[0,-1]
y_pred_final = best_clf.predict(test_df[feature_columns])

# Output to file for submission
submission = pd.DataFrame({'PassengerId': test_PassengerId, 'Survived': y_pred_final})
submission.to_csv('submission 8 - Tuned DT.csv', index=False)


