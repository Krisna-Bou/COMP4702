import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

regr = pd.read_csv("./w3classif.csv")
knn = KNeighborsClassifier(n_neighbors=1)

x = regr.columns.tolist()[0]
y = regr.columns.tolist()[1]
z = regr.columns.tolist()[2]



knn.fit(list(zip(regr[x], regr[y])), regr[z])

new_x = 8
new_y = 21
new_point = [(new_x, new_y)]

prediction = knn.predict(new_point)

# Creates scatter plot of classification csv file. 
plt.title(f"Scatter Plot - Classification")
plt.scatter(regr[x] + [new_x], regr[y] + [new_y], c=regr[z] + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()
