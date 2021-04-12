import sys
import numpy as np
import pandas as pd
import sklearn
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


def process_data(data_raw):
    result = []
    for column in data_raw:
        if column != "adm_id":
            df = pd.get_dummies(data_raw[column])
            result.append(df)
        else:
            result.append(data_raw[column])
    sparse_matrix = pd.concat(result, axis=1)
    return sparse_matrix


def accuracy(predictions, target):
    counter = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = predictions.shape[0]
    for x in range(predictions.shape[0]):
        print(predictions[x])
        if predictions[x] == target[x]:
            counter += 1
            if predictions[x] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predictions[x] == 1:
                fp += 1
            else:
                fn += 1
    return counter / total, (tp/ (tp + fp)), (tp/ (tp + fn)), (tn/ (tn + fp))


features = pd.read_csv(sys.argv[1])
demos = process_data(pd.read_csv(sys.argv[2]))
labels = pd.read_csv(sys.argv[3])


df_merge1=pd.merge(features, demos, on='adm_id')
df_merge = pd.merge(df_merge1, labels, on='adm_id')
print(df_merge.shape)
df_merge.dropna(thresh=int(.7 * df_merge.shape[0]),inplace=True, axis=1)
df_merge = df_merge.fillna(0)
print(df_merge.shape)
print(df_merge)

# output labels
y = df_merge['mortality']


#Split data into test and train
"""x = df_merge.drop(['mortality', 'adm_id'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.65, random_state=0)
regressor = LogisticRegression(max_iter=10000)
regressor.fit(x, y) #training the algorithm
predictions = regressor.predict(x)
#a = regressor.score(x_test, y_test)
#print(a)
a, p, r, s = accuracy(predictions, y.to_numpy())
print("Accuracy: {}\nPrecision: {}\nRecall: {}\nSpec: {}\n".format(a, p, r, s))"""

x = pd.merge(features, labels, on="adm_id")
x.dropna(thresh=int(.7 * x.shape[0]), axis = 1)
x = x.fillna(0)
y = x['mortality']
x.drop(['mortality', 'adm_id'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.65, random_state=0)

gnb = GaussianNB()
gnb.fit(x_train, y_train)
predictions = gnb.predict(x_test)
a, p, r, s = accuracy(predictions, y_test.to_numpy())
print("Accuracy: {}\nPrecision: {}\nRecall: {}\nSpec: {}\n".format(a, p, r, s))