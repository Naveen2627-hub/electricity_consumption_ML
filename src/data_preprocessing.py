import pandas as pd
import numpy as np 
import pickle
import logging
from sklearn.model_selection import train_test_split

def remove_outliers(df):

    cols = [col for col in df.columns if col not in ['Date','Time']]
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3-q1
        df = df[(df[col] >= (q1 - 1.5*iqr)) & (df[col] <= (q3 + 1.5*iqr))]
    return df

def removing_nulls(df):

    #Relacing '?' with mean of that columns
    cols = [col for col in df.columns if col not in ['Date','Time']]
    df[cols] = df[cols].replace('?',np.nan)
    df[cols] = df[cols].apply(pd.to_numeric, errors= 'coerce')
    df[cols] = df[cols].fillna(df[cols].mean())
    return df

def split_data(df):
    
    x = df.iloc[:,3:-1]
    y = df.iloc[:,2:3]
    x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test

def preprocess(df):

    #Replacing NULL
    df = removing_nulls(df)
    #Removing outliers
    df = remove_outliers(df)
    #Split data into Test and train
    x_train, x_test, y_train, y_test = split_data(df)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    #input path
    input_data_path = r"C:\Users\navee\OneDrive\Documents\Courses\Projects\electricity_consumption_ML\data\household_power_consumption.txt"

    #Read data
    df = pd.read_csv(input_data_path, delimiter=";")
    df = df.head(10000).copy()

    x_train, x_test, y_train, y_test = preprocess(df)

    x_train.to_csv(r"C:\Users\navee\OneDrive\Documents\Courses\Projects\electricity_consumption_ML\data\cleaned\x_train.csv", index=False)
    x_test.to_csv(r"C:\Users\navee\OneDrive\Documents\Courses\Projects\electricity_consumption_ML\data\cleaned\x_test.csv", index=False)
    y_train.to_csv(r"C:\Users\navee\OneDrive\Documents\Courses\Projects\electricity_consumption_ML\data\cleaned\y_train.csv", index=False)
    y_test.to_csv(r"C:\Users\navee\OneDrive\Documents\Courses\Projects\electricity_consumption_ML\data\cleaned\y_test.csv", index=False)


    

