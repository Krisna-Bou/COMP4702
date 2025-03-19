import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

regr = pd.read_csv("./w3regr.csv")
knn = KNeighborsClassifier(n_neighbors=1)

x = regr.columns.tolist()[0]
y = regr.columns.tolist()[1]

knn.fit(regr)

new_x = 8
new_y = 21
new_point = [(new_x, new_y)]

prediction = knn.predict(new_point)

plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.title(f"Scatter Plot - Classification")
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()
