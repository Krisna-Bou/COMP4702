import pandas as pd
import matplotlib.pyplot as plt

regr = pd.read_csv("./w3regr.csv").sample(frac=1).reset_index(drop=True)

column_x = regr.columns.tolist()[0]
column_y = regr.columns.tolist()[1]

#Creates scatter plot of regression csv file. 
plt.scatter(regr[column_x], regr[column_y], alpha=0.5)
plt.title(f"Scatter Plot - Regression")
plt.show()

regr = pd.read_csv("./w3classif.csv").sample(frac=1).reset_index(drop=True)

column_x = regr.columns.tolist()[0]
column_y = regr.columns.tolist()[1]

#Creates scatter plot of classification csv file. 
plt.scatter(regr[column_x], regr[column_y], alpha=0.5)
plt.title(f"Scatter Plot - Classification")
plt.show()
