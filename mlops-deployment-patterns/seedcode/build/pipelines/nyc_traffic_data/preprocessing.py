import argparse
import os
import requests
import tempfile
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

feature_columns_dtype = {
    "BOROUGH": str,
    "TIME": str,
    "WEEKDAY": str
}
label_column_dtype = {"NUMBER OF PERSONS INJURED": int}

all_column_types = {
    "BOROUGH": str,
    "TIME": str,
    "WEEKDAY": str,
    "NUMBER OF PERSONS INJURED": int,
    "NUMBER OF PERSONS INJURED": int,
}


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    input_data_uri = 's3://sagemaker-ml-ops-utility-bucket/sm_pipeline_template/training_data.csv'

    # df = pd.read_csv(
    #     f"{base_dir}/input/training_data.csv",
    #     dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype)
    # )
    df = pd.read_csv(
        input_data_uri,
        dtype=all_column_types
    )

    categorical_features = ['BOROUGH' ,'TIME', 'WEEKDAY']
    
    df = df.dropna()
    
    df = pd.get_dummies(df, columns=categorical_features)
    df = df.loc[:,df.columns[::-1]]
    
    df = df.sample(frac=1).reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    
    train_df.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    val_df.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test_df.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    