# You can use Caret with R or any other library. You can use scikit-learn with Python or the code from scratch. 

# 1. Create 60/40 train test spilt and report training and test performance in terms of classification performance.
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import linear_model

import matplotlib.pyplot as plt

DIABETES_PATH = "./week_1/exercise_1_5/"

def load_diabetes_data(data_path = DIABETES_PATH):
    csv_path = os.path.join(data_path, "pima-indians-diabetes.csv")
    return pd.read_csv(csv_path, header = None, names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Class"])

dia_data = load_diabetes_data()

train_set, test_set = train_test_split(dia_data, test_size = 0.4, random_state = 42)

dia_data['Class'].value_counts()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
for train_index, test_index in split.split(dia_data, dia_data['Class']):
    strat_train_set = dia_data.loc[train_index]
    strat_test_set = dia_data.loc[test_index]

# train_set['Class'].value_counts()/len(train_set)
# strat_train_set['Class'].value_counts()/len(strat_train_set)
# dia_data['Class'].value_counts()/len(dia_data)

def log_reg_scipy(x_train, x_test, y_train, y_test):
    logistic = linear_model.LogisticRegression(solver='lbfgs', C=1e5, multi_class='multinomial')
    logistic.fit(x_train, y_train)

    # print(logistic.coef_)

    print(logistic.coef_) 
    error = np.mean((logistic.predict(x_test) - y_test)**2) 
    print(error, 'is error') 
    r_score = logistic.score(x_test, y_test)  # Explained variance score: 1 is perfect prediction
                                                  # and 0 means that there is no linear relationshipbetween X and y.

    print(r_score, 'is R score')

strat_x_train = strat_train_set.loc[:, strat_train_set.columns != 'Class']
strat_y_train = strat_train_set.loc[:, strat_train_set.columns == 'Class']
strat_x_test = strat_test_set.loc[:, strat_test_set.columns != 'Class']
strat_y_test = strat_test_set.loc[:, strat_test_set.columns == 'Class']

log_reg_scipy(strat_x_train, strat_x_test, strat_y_train, strat_y_test)

# 2. Report AUC and ROC and Precision-Recall curve, F1 Score. 

# 3. Try L1/L2 and elastic net regularisation and compare results. 

# 4.Carry out 10 independent experiment runs for each case and report the mean and std of the results in Part 2.

# 5. Carry out 10-fold cross-validation on the original dataset and report the results required in Part 2. 