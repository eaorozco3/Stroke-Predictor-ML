import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as kernel_lib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from dataset import clean_dataset, categorical_to_numerical
from perceptron import regular_perceptron

if __name__ == "__main__":
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df_updated = clean_dataset(df)
    print(df_updated)

    x = df_updated.drop("stroke", axis = 1)
    y = df_updated["stroke"]
    x_numerical = categorical_to_numerical(x.copy())

    X_train, X_test, y_train, y_test = train_test_split(x_numerical, y, test_size=0.2, random_state=42, stratify=y)
    #print(x_numerical)
    regular_perceptron = regular_perceptron(X_train, y_train)
    y_pred = regular_perceptron.predict(X_test)
    print("Regular Perceptron Results:")
    print(classification_report(y_test, y_pred))

