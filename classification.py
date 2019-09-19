from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from mlxtend.classifier import StackingCVClassifier
import pandas as pd
from utils.feature import feature_eng
from utils.detect import missing_values
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import numpy as np

X = pd.read_csv('/Users/ricklicona/PycharmProjects/Classification_of_Forest_Types_Challenge/data/train.csv', index_col='Id')
X_test_full = pd.read_csv('/Users/ricklicona/PycharmProjects/Classification_of_Forest_Types_Challenge/data/test.csv', index_col='Id')

# SEPARATE DATA AND LABELS

X.dropna(axis=0, subset=['Cover_Type'], inplace=True)
y = X.Cover_Type
X.drop(['Cover_Type'], axis=1, inplace=True)


# VERIFY THE EXISTENCE OF MISSING VALUES
missing_values(X, X_test_full)


# APPLYING FEATURE ENGINEERING TO DATA.
# WE ELIMINATE THE Soil_Type7' AND 'Soil_Type15' FEATURES DUE TO THE POOR CORRELATION WITH THE REST OF FEATURES
X = feature_eng(X)
X_test_full = feature_eng(X_test_full)


# STACKING USING RANDOM FOREST AND EXTRA TREE CLASSIFIER

random_state = 42
n_jobs = -1
rf2_clf = RandomForestClassifier(n_estimators=719,
                                 max_features=0.3,
                                 max_depth=464,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 bootstrap=False,
                                 random_state=42,
                                 n_jobs=-1)

rf_clf = RandomForestClassifier(n_estimators=400,
                                min_samples_leaf=1,
                                verbose=0,
                                random_state=random_state,
                                n_jobs=n_jobs)

ex_cls = ExtraTreesClassifier(n_estimators=700, criterion='entropy', min_samples_split=3, random_state=42,
                              max_features=0.3,
                              max_depth=464,
                              min_samples_leaf=1,
                              n_jobs=-1)

ensemble = [('ex_cls', ex_cls),
            ('rf2', rf2_clf),
            ('rf', rf_clf)]

stack = StackingCVClassifier(classifiers=[clf for label, clf in ensemble],
                             meta_classifier=rf_clf,
                             cv=5,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1)
# HOLD-OUT
X_train, X_valid, y_train, y_valid = train_test_split(X.values, y.values, train_size=0.8, test_size=0.2,
                                                      random_state=42)

stack = stack.fit(X_train, y_train)
pr = stack.predict(X_valid)

# MAE
y_nump = np.array(y_valid)
mae = mean_absolute_error(pr, y_valid)
print("Mean Absolute Error:", mae)
print("Good predicted: ", np.sum(pr == y_nump), "of: ", y_valid.shape[0])
print("Accuracy Score: ", accuracy_score(pr, y_valid)*100)

# PREDICTION OF TEST
pr_final_test = stack.predict(X_test_full)
print(pr_final_test.shape)


