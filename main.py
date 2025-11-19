import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
    r_perceptron = regular_perceptron(X_train, y_train)
    y_pred = r_perceptron.predict(X_test)
    print("\nRegular Perceptron Results:")
    print(classification_report(y_test, y_pred))

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)
    scaled_perceptron = regular_perceptron(x_train_scaled, y_train)
    y_scaled_pred = scaled_perceptron.predict(x_test_scaled)
    print("\nScaled Perceptron Results:")
    print(classification_report(y_test, y_scaled_pred))
