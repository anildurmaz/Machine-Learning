import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Healthy gearbox
# ---------------
h30hz0  = pd.read_csv("Healthy/h30hz0.csv")
h30hz10 = pd.read_csv("Healthy/h30hz10.csv")
h30hz20 = pd.read_csv("Healthy/h30hz20.csv")
h30hz30 = pd.read_csv("Healthy/h30hz30.csv")
h30hz40 = pd.read_csv("Healthy/h30hz40.csv")
h30hz50 = pd.read_csv("Healthy/h30hz50.csv")
h30hz60 = pd.read_csv("Healthy/h30hz60.csv")
h30hz70 = pd.read_csv("Healthy/h30hz70.csv")
h30hz80 = pd.read_csv("Healthy/h30hz80.csv")
h30hz90 = pd.read_csv("Healthy/h30hz90.csv")

# Broken gearbox
# --------------
b30hz0  = pd.read_csv("BrokenTooth/b30hz0.csv")
b30hz10 = pd.read_csv("BrokenTooth/b30hz10.csv")
b30hz20 = pd.read_csv("BrokenTooth/b30hz20.csv")
b30hz30 = pd.read_csv("BrokenTooth/b30hz30.csv")
b30hz40 = pd.read_csv("BrokenTooth/b30hz40.csv")
b30hz50 = pd.read_csv("BrokenTooth/b30hz50.csv")
b30hz60 = pd.read_csv("BrokenTooth/b30hz60.csv")
b30hz70 = pd.read_csv("BrokenTooth/b30hz70.csv")
b30hz80 = pd.read_csv("BrokenTooth/b30hz80.csv")
b30hz90 = pd.read_csv("BrokenTooth/b30hz90.csv")

Failure = 1
h30hz0['Failure'] = np.ones((len(h30hz0.index),1))

Failure = 0
b30hz0['Failure'] = np.zeros((len(b30hz0.index),1))

df = pd.concat([b30hz0,h30hz0], axis=0)
print(df)

sns.heatmap(np.abs(df.corr()),annot=True,cmap='cubehelix_r')

plt.show()