import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_df = pd.read_csv(os.path.join(BASE_DIR, "datasets/train.csv"))
train_df.fillna(-999,inplace=True)
train_df["Sex"] = train_df["Sex"].replace({'male':1, 'female':0})
train_df['Fare_Category'] = pd.cut(train_df['Fare'], bins=[0,7.90,14.45,31.28,120], labels=['Low','Mid', 'High_Mid','High'])
most_frequent_value_train = train_df['Fare_Category'].mode()[0]
train_df['Fare_Category'] = train_df['Fare_Category'].fillna(most_frequent_value_train)
features_train = ['Survived', 'Pclass', 'Sex', 'Age', 'Parch', 'Fare_Category']
train_df[features_train].to_csv(os.path.join(BASE_DIR, "datasets/train_prep.csv"), columns=features_train)