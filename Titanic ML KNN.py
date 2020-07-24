import numpy as np
import pandas as pd
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


# ML Modelling starts here ->
# Let's experiment with KNN
# Perform hyperparameter tuning using Gridsearch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score

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

knn_param_grid = {
    'n_neighbors': np.arange(2,10,1),
    'leaf_size': np.arange(5,50,5),
    'p': np.arange(1.0,2.0, 0.25)
}

# Using ROC AUC as the scoring mechanism
knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid=knn_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
knn_cv.fit(X_train, y_train)

print('Parameters tested :', knn_param_grid)
print('Best Parameters: ', knn_cv.best_params_)
print('Best Score: ', knn_cv.best_score_)

y_pred = knn_cv.predict(X_test)
y_pred_proba = knn_cv.predict_proba(X_test)[:, 1]
print('Accuracy Score on Validation: ', accuracy_score(y_test, y_pred))
print('ROC AUC Score on Validation: ', roc_auc_score(y_test, y_pred_proba))

# Use output of above to setup a new LR on the full train dataset & predict final output

knn = KNeighborsClassifier(leaf_size=20, n_neighbors=7, p=1)
knn.fit(X,y)
y_pred_final = knn.predict(test_df[feature_columns])
submission = pd.DataFrame({'PassengerId': test_PassengerId, 'Survived': y_pred_final})
submission.to_csv('submission 6 - KNN.csv', index=False)

