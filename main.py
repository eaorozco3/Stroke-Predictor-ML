import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as kernel_lib
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScale
from dataset import clean_dataset

if __name__ == "__main__":
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df_updated = clean_dataset(df)
    print(df_updated)

    x = df_updated.drop("stroke", axis = 1)
    y = df_updated["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

