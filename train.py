import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# file input
df = pd.read_csv("data/train.csv")
df = df[0:1500]
X = df.drop(columns=[[0, 1, 2, 3, 4, 5]], axis = 1).to_numpy()
y = df['Disease'].to_numpy()
labels = np.unique(y)
y = np.array([np.where(labels == x) for x in y]).flatten()

model = LogisticRegression().fit(X, y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
