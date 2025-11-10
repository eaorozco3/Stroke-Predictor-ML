def clean_dataset(df):
    df_updated = df.copy()

    df_updated = df_updated.drop("id", axis = 1)
    df_updated["bmi"] = df_updated["bmi"].fillna(df_updated["bmi"].mean())

    return df_updated