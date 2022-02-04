from time import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')


def model_accuracy(model, data):
    '''
    Retuns a accuracy, mean absolute error, mean squared error and root mean squared error of the model object passed in 'model'

    model : It is the object of the model to be tested

    data : It should be a list of xtrain, ytrain, xtest, ytest

    models_obj : Boolean value determines if to return the model and predicted value
    '''
    xtrain = data[0]
    ytrain = data[2]
    xtest = data[1]
    ytest = data[3]
    model_return = model
    model.fit(xtrain, ytrain)
    return model    


def load_model():
    start = time()
    print(start)
    df = pd.read_csv('cleanData.csv')
    df.drop('Unnamed: 0',axis=1,inplace=True)
    x = df.drop('price', axis=1)
    y = df.price
    encoders = {}
    for i in x.drop('odometer',axis=1).columns:
        le = LabelEncoder()
        encoders[i] = le.fit(x[i])
    le = LabelEncoder()
    x[x.drop('odometer',axis=1).columns] = x[x.drop('odometer',axis=1).columns].apply(le.fit_transform)
    x = x.values
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2, random_state=42)
    td1 = [xtrain, xtest, ytrain, ytest]
    model = model_accuracy(    BaggingRegressor(xgb(n_estimators=10, learning_rate=.3, max_depth=10, gamma=0, subsample=.9), random_state=42,
                         n_jobs=2), td1)
    # model = model_accuracy(
    #     BaggingRegressor(xgb(n_estimators=1000, learning_rate=.3, max_depth=10, gamma=0, subsample=.9), random_state=42,
    #                      n_jobs=2),td1)              #better accuracy

    # model = BaggingRegressor(xgb(n_estimators=2500, learning_rate=.3, max_depth=10, gamma=0, subsample=.9),
    #                          random_state=42,
    #                          n_jobs=2)                          #increased accuracy at the cost of computational time

    # model = model_accuracy(BaggingRegressor(RandomForestRegressor(n_estimators=100, random_state=0), random_state=42, n_jobs=-1),td1)
    # model = model_accuracy(BaggingRegressor(RandomForestRegressor(n_estimators=20, random_state=0), random_state=42, n_jobs=-1),td1)
    
    end = time()
    print(end)
    print(end-start)
    return model, encoders, df
