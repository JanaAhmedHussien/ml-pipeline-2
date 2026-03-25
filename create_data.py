from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

import os
os.makedirs("data", exist_ok=True)

df.to_csv("data/train.csv", index=False)