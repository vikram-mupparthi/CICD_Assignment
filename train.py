import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle
import numpy as np

# file input
df = pd.read_csv("data/train.csv")
df = df.sample(frac=0.1, random_state=42)
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
model = DecisionTreeClassifier(max_depth=2)  # Using DecisionTreeClassifier with limited depth
model.fit(X, y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
