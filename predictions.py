import sys
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder



def count_nan(tuple):
    count = 0
    for x in tuple:
        if math.isnan(float(x)):
            count += 1
    return count


def get_best(dataframe):
    better = dataframe.values.tolist()
    min = 1000
    min_index = 0
    for i in range(len(better)):
        check = count_nan(better[i])
        if check < min:
            min = check
            min_index = i
    return better[min_index]


def process_features(features):
    processed_features = []
    column_names = list(features.columns)
    for x in features['adm_id'].unique().tolist():
        processed_features.append(get_best(features.loc[features['adm_id'] == x]))
    return pd.DataFrame(processed_features, columns=column_names)


def read_files():
    features = pd.read_csv(sys.argv[1])
    features = process_features(features)
    demos = pd.read_csv(sys.argv[2])
    labels = pd.read_csv(sys.argv[3])
    return features, demos, labels


def get_xy():
    features, demos, labels = read_files()
    enc = OrdinalEncoder()
    x = pd.merge(features, labels, on="adm_id")
    x.dropna(thresh=int(.7 * x.shape[0]), axis=1)
    x = x.fillna(0)
    x = pd.merge(demos, x, on="adm_id")
    #adam_x = x['adm_id']
    y = x['mortality']
    x = x.drop(['mortality', 'adm_id'], axis=1)
    x[["gender", "age", "admission_type", "admission_location", "insurance", "marital_status",
       "ethnicity"]] = enc.fit_transform(
        x[["gender", "age", "admission_type", "admission_location", "insurance", "marital_status", "ethnicity"]])
    return x, y


def train(x, y):
    model = GaussianNB()
    model.fit(x,y)
    return model


def test(x, model):
    return model.predict_proba(x)


def train_k_models(k):
    X, Y = get_xy()
    all_models = []
    for i in range(k):
        xtrain, tmp1, ytrain, tmp2 = train_test_split(X, Y, test_size=.7)
        all_models.append(train(xtrain, ytrain))
    return all_models


def individual_predictions(x, model):
    prob = test(x, model)
    return prob


def test_data():
    features = pd.read_csv(sys.argv[4])
    features = process_features(features)
    demos = pd.read_csv(sys.argv[5])
    adm = demos['adm_id']
    enc = OrdinalEncoder()
    x = pd.merge(features, demos, on="adm_id")
    x.dropna(thresh=int(.7 * x.shape[0]), axis=1)
    x = x.fillna(0)
    x = x.drop(['adm_id'], axis=1)
    x[["gender", "age", "admission_type", "admission_location", "insurance", "marital_status",
       "ethnicity"]] = enc.fit_transform(
        x[["gender", "age", "admission_type", "admission_location", "insurance", "marital_status", "ethnicity"]])
    return x, adm


def print_csv(prob, adm):
    final = []
    a = adm.values.tolist()
    p = prob.tolist()
    for i in range(len(p)):
        final.append([a[i],p[i][1]])
    file = pd.DataFrame(final, columns=['adm_id', 'probability'])
    file.to_csv("Results.csv", index=False)


def total_1(p):
    pp = p.tolist()
    count = 0
    for i in range(len(pp)):
        if pp[i][1] > .5:
            count += 1
    print( count / len(pp))


def print_predictions(models):
    x, adm = test_data()
    prob = individual_predictions(x, models[0])
    for i in range(1, len(models)):
        prob += individual_predictions(x, models[i])
    prob /= len(models)
    total_1(prob)
    print_csv(prob, adm)


k = 4
models = train_k_models(k)
print_predictions(models)

