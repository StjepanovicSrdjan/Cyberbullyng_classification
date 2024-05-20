from load_dataset import get_embedding_data, get_tf_idf
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier


def train_and_eval_svm(X_train, y_train, x_test, y_test, save_model=False):

    # classifier = svm.SVC()
    # parameters = {'C': [0.1, 1, 10, 100, 1000],
    #             'kernel': ['linear', 'rbf']}
    # kf = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)
    # gs_clf = GridSearchCV(classifier, parameters, cv=kf)
    # gs_clf = gs_clf.fit(X_train, y_train)

    # print("SVM best parameters: ")
    # print(gs_clf.best_params_)

    classifier = SVC(kernel="rbf")
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(x_test)

    print(classification_report(y_test, predictions))

    if save_model:
        with open("svm.pickle", "wb") as file:
            pickle.dump(classifier, file)


def train_and_eval_xgboost(X_train, y_train, x_test, y_test, save_model=False):

    # param_grid = {
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.1, 0.01, 0.001],
    #     'subsample': [0.5, 0.7, 1]
    # }

    # xgb_model = xgb.XGBClassifier()

    # gs_clf = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')

    # gs_clf.fit(X_train, y_train)

    # print("xgbOOST best parameters: ")
    # print(gs_clf.best_params_)


    classifier = xgb.XGBClassifier(n_estimators=100, subsample=0.7, colsample_bytree=0.6, random_state=123)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(y_test, predictions)
    print(classification_report(y_test, predictions))

    if save_model:
            with open("xgb.pickle", "wb") as file:
                pickle.dump(classifier, file)

def train_and_eval_rf(X_train, y_train, x_test, y_test, save_model=False):

    # parameters = {
    #             "n_estimators": [10, 50, 100],
    #             "criterion": ["gini", "entropy"],
    #             "bootstrap": [False],
    #             "warm_start": [True],
    #             "random_state": [123]
    #             }

    # kf = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)
    # gs_clf = GridSearchCV(classifier, parameters, cv=kf)
    # gs_clf = gs_clf.fit(X_train, y_train)

    # print("RF best parameters: ")
    # print(gs_clf.best_params_)

    # validation_score = gs_clf.score(X_val, y_val)
    # print(validation_score)


    classifier = RandomForestClassifier(bootstrap=False, criterion='gini', n_estimators= 100, random_state= 123, warm_start= True)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(y_test, predictions)
    print(classification_report(y_test, predictions))

    if save_model:
            with open("rbf.pickle", "wb") as file:
                pickle.dump(classifier, file)


def train_and_eval_bagging(X_train, y_train, x_test, y_test, save_model=False):

    svm = SVC(kernel='rbf')
    classifier = BaggingClassifier(estimator=svm, n_estimators=10, random_state=123)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(y_test, predictions)
    print(classification_report(y_test, predictions))

    if save_model:
            with open("rbf.pickle", "wb") as file:
                pickle.dump(classifier, file)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = get_embedding_data()
    #  X_train, y_train, X_val, y_val, X_test, y_test = get_tf_idf()
    
    train_and_eval_svm(X_train, y_train, X_test, y_test)
    # train_and_eval_rf(X_train, y_train, X_test, y_test)
    # train_and_eval_xgboost(X_train, y_train, X_test, y_test)
    # train_and_eval_bagging(X_train, y_train, X_test, y_test)