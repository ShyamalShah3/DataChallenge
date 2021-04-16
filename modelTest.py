import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt


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


def read_files(f = False, d = False, l = False):
    features, demos, labels = None, None, None
    if f:
        features = pd.read_csv(sys.argv[1])
    if d:
        demos = process_data(pd.read_csv(sys.argv[2]))
    if l:
        labels = pd.read_csv(sys.argv[3])
    return features, demos, labels


def all_split():
    features, tmp2, labels = read_files(f=True, l=True)
    enc = OrdinalEncoder()
    x = pd.merge(features, labels, on="adm_id")
    x.dropna(thresh=int(.3 * x.shape[0]), axis=1)
    x = x.fillna(0)
    x = pd.merge(pd.read_csv(sys.argv[2]), x, on="adm_id")
    y = x['mortality']
    x = x.drop(['mortality', 'adm_id'], axis=1)
    x[["gender", "age", "admission_type", "admission_location", "insurance", "marital_status",
       "ethnicity"]] = enc.fit_transform(
        x[["gender", "age", "admission_type", "admission_location", "insurance", "marital_status", "ethnicity"]])
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.7)
    return x_train, x_val, y_train, y_val


def feature_and_target_split():
    features, tmp, labels = read_files(f=True, l=True)
    x = pd.merge(features, labels, on="adm_id")
    x.dropna(thresh=int(.7 * x.shape[0]), axis=1)
    x = x.fillna(0)
    y = x['mortality']
    x = x.drop(['mortality', 'adm_id'], axis=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.7)
    return x_train, x_val, y_train, y_val


def demos_and_target_split():
    tmp1, tmp2, labels = read_files(l=True)
    enc = OrdinalEncoder()
    x = pd.merge(pd.read_csv(sys.argv[2]), labels, on="adm_id")
    y = x['mortality']
    x = x.drop(['mortality', 'adm_id'], axis=1)
    x[["gender", "age", "admission_type", "admission_location", "insurance", "marital_status",
       "ethnicity"]] = enc.fit_transform(
        x[["gender", "age", "admission_type", "admission_location", "insurance", "marital_status", "ethnicity"]])
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.7, random_state=0)
    return x_train, x_val, y_train, y_val


def accuracy(predictions, target):
    counter = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = predictions.shape[0]
    for x in range(predictions.shape[0]):
        #print(predictions[x])
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


def train_log_regression(x, y):
    reg = LogisticRegression(max_iter=10000)
    reg.fit(x, y)  # training the algorithm
    return reg


def train_gaussianNB(x, y):
    gnb = GaussianNB()
    gnb.fit(x, y)
    return gnb


def train_categoricalNB(x, y):
    cat = CategoricalNB()
    cat.fit(x, y)
    return cat


def test_model(x, y, model):
    predictions = model.predict(x)
    predicted_prob = model.predict_proba(x)
    a, p, r, s = accuracy(predictions, y.to_numpy())
    print("Accuracy: {}\nPrecision: {}\nRecall: {}\nSpec: {}\n\n\n".format(a, p, r, s))
    return predictions, predicted_prob


def plot_roc(predictions, targets, title):
    fpr1, tpr1, threshold1 = roc_curve(targets, predictions[:,1])
    roc_auc1 = auc(fpr1,tpr1)
    plt.plot(fpr1,tpr1, 'b', label='Experiment 1 AOC = %0.2f' % roc_auc1)
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('{} ROC Curve'.format(title))
    plt.show()
    return


def plot_all(target_rg, target_cat, title, reg, gnb, cat):
    fpr1, tpr1, threshold1 = roc_curve(target_rg, reg[:, 1])
    roc_auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, 'b', label='Logistic Regression AOC = %0.2f' % roc_auc1)

    fpr1, tpr1, threshold1 = roc_curve(target_rg, gnb[:, 1])
    roc_auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, 'r', label='Gaussian NB AOC = %0.2f' % roc_auc1)

    fpr1, tpr1, threshold1 = roc_curve(target_cat, cat[:, 1])
    roc_auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, 'g', label='Categorical NB Regression AOC = %0.2f' % roc_auc1)
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('{} ROC Curve'.format(title))
    plt.show()
    return


def main(reg = False, gnb = False, cat = False):
    reg_model, gnb_model, cat_model = None, None, None
    reg_prob, gnb_prob, cat_prob = None, None, None
    yval_rg, yval_cat = None, None
    if reg or gnb:
        xtrain, xval, ytrain, yval_rg = all_split()
        if reg:
            reg_model = train_log_regression(xtrain, ytrain)
            predictions, reg_prob = test_model(xval, yval_rg, reg_model)
            plot_roc(reg_prob, yval_rg, "Regression")
        if gnb:
            gnb_model = train_gaussianNB(xtrain, ytrain)
            predictions, gnb_prob = test_model(xval, yval_rg, gnb_model)
            plot_roc(gnb_prob, yval_rg, "Gaussian NB")
    if cat:
        xtrain, xval, ytrain, yval_cat = demos_and_target_split()
        cat_model = train_categoricalNB(xtrain, ytrain)
        predictions, cat_prob = test_model(xval, yval_cat, cat_model)
        plot_roc(cat_prob, yval_cat, "Categorical NB")
    if reg and gnb and cat:
        plot_all(yval_rg, yval_cat, "All", reg_prob, gnb_prob, cat_prob)


main(reg = True, gnb=True, cat=True)