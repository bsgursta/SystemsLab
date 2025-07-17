import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('train.csv')
timeseries = df[["sales"]]

plt.plot(timeseries)
plt.show()