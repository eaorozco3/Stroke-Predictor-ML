import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as kernel_lib
from dataset import clean_dataset

if __name__ == "__main__":
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df_updated = clean_dataset(df)
    print(df_updated)
