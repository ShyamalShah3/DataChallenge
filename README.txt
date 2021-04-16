You will need to download the following packages to run our code:

import sys
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
...


The zip file contains the following folders and files:

    DataSet: Folder that holds the input data

    Documentation: Folder that contains the Report, the README file, as well as the Results

    modelTest.py: This file was used to test and compare the different models

    predictions.py: This file was used to train the selected model and output a csv file called "Results.csv" with the calculated predictions


To run modelTest.py use the following command:

    python modelTest.py Data/train_features.csv Data/train_demos.csv Data/train_labels.csv

    The input files' path can be replaced but the order MUST remain the same, otherwise the code will not run properly.
    This file output 4 different ROC curves showing the results of training the different models with the training and
    validation data.

To run predictions.py use the following command:

    python predictions.py Data/train_features.csv Data/train_demos.csv Data/train_labels.csv Data/test_features.csv Data/test_demos.csv

    The input files' path can be replaced but the order MUST remain the same, otherwise the code will not run properly.
    This file outputs a csv file called "Results.csv" that contains the adm_id of the patient and the probability for
    the patient’s in-mortality.