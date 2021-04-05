# %% [code]
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn 
import numpy as np
 
# %% [code]
np.random.seed(10)

# %% [code]
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")

# %% [code]
df.head()

# %% [code]
x = df.drop("DEATH_EVENT",axis=1)
y = df["DEATH_EVENT"]

# %% [code]
print(x,y)

# %% [code]
from sklearn.linear_model import LogisticRegression

# %% [code]
model = LogisticRegression().fit(x,y)

# %% [code]
model 

# %% [code]
model.score(x,y)

# %% [markdown]
# as of now no conflicts and 82% accuracy not good neither bad. now shall save the model and with that we end our day 6 of 100 days of code now. hope you enjoyed using the code file. it was great coding this solution. Now shall take a break.

# %% [code]
import pickle as pic 

# %% [code]
with open("model.pkl","w") as f:
    f.write(str(pic.dumps(model)))
    f.close()

# %% [code]
pic.dumps(model)

# %% [markdown] 
# 0 conflicts many happiness just love it work is done.