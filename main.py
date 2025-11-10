import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as kernel_lib

if __name__ == "__main__":
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    print(df)
