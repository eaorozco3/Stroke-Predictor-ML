import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from dataset import clean_dataset, categorical_to_numerical, z_score_normalization
from perceptron import regular_perceptron
from random_forest import generate_rf
from graphsage_model import generate_graphsage_embeddings
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df_updated = clean_dataset(df)
    df_updated = z_score_normalization(df_updated)
    print(df_updated)

    x = df_updated.drop("stroke", axis = 1)
    y = df_updated["stroke"]
    x_numerical = categorical_to_numerical(x.copy())

    X_train, X_test, y_train, y_test = train_test_split(x_numerical, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)


    r_perceptron = regular_perceptron(X_train, y_train)
    y_pred = r_perceptron.predict(X_test)
    print("\nRegular Perceptron Results:")
    print(classification_report(y_test, y_pred))
    f1_perceptron = f1_score(y_test, y_pred)


    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)
    scaled_perceptron = regular_perceptron(x_train_scaled, y_train)
    y_scaled_pred = scaled_perceptron.predict(x_test_scaled)
    print("\nScaled Perceptron Results:")
    print(classification_report(y_test, y_scaled_pred))
    f1_scaled = f1_score(y_test, y_scaled_pred)

    rf = generate_rf(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("\nRandom Forest Results:")
    print(classification_report(y_test, y_pred_rf))
    f1_rf = f1_score(y_test, y_pred_rf)    

    print("\nGenerating GraphSAGE embeddings...")

    emb_train = generate_graphsage_embeddings(X_train)
    emb_test = generate_graphsage_embeddings(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(emb_train, y_train)

    y_pred_sage = clf.predict(emb_test)
    print("\nGraphSAGE + Logistic Regression Results:")
    print(classification_report(y_test, y_pred_sage))

    f1_sage = f1_score(y_test, y_pred_sage)

    # --------------------------------------------------------
    # Comparison Chart
    # --------------------------------------------------------
    models = ["Perceptron", "Scaled Perceptron", "Random Forest", "GraphSAGE"]
    scores = [f1_perceptron, f1_scaled, f1_rf, f1_sage]

    plt.figure(figsize=(10, 5))
    plt.bar(models, scores)
    plt.title("Model Comparison for Stroke Prediction")
    plt.ylabel("F1-score")
    plt.show()