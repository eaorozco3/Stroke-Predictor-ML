from sklearn.calibration import LabelEncoder

def clean_dataset(df):
    df_updated = df.copy()

    df_updated = df_updated.drop("id", axis = 1)
    df_updated["bmi"] = df_updated["bmi"].fillna(df_updated["bmi"].mean())

    return df_updated

def categorical_to_numerical(x):
    categorical_columns = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

    for feature in categorical_columns:
        encoder = LabelEncoder()
        x[feature] = encoder.fit_transform(x[feature])
    
    return x 

def z_score_normalization(x):
    numerical_columns = ["age", "avg_glucose_level", "bmi"]

    for feature in numerical_columns:
        mean = x[feature].mean()
        std = x[feature].std()
        x[feature] = (x[feature] - mean) / std
    
    return x