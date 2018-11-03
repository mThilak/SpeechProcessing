import scipy
import numpy as np
import matplotlib
import pandas
import sklearn
from matplotlib import pyplot
import matplotlib.pyplot as plt
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
print(dataset.head(150))
dataset.groupby('class').hist()
plt.show()
#array_name = np.array(dataset_name.[beginnning_index:end_index, dimension])
sl_setosa = np.array(dataset.iloc[0:50, 0])
pyplot.hist(sl_setosa, alpha = 0.5, label = 'sl_setosa')
sl_versicolor = np.array(dataset.iloc[50:100,0])
pyplot.hist(sl_versicolor, alpha = 0.5, label = 'sl_versicolor')
sl_virginica = np.array(dataset.iloc[100:150,0])
pyplot.hist(sl_virginica, alpha = 0.5, label = 'sl_virginica')
plt.xlabel("Sepal Length")
plt.ylabel("No.of Flowers")
plt.legend()
plt.show()

sw_setosa = np.array(dataset.iloc[0:50,1])
pyplot.hist(sw_setosa, alpha = 0.5, label = 'sw_setosa')
sw_versicolor = np.array(dataset.iloc[50:100, 1])
pyplot.hist(sw_versicolor, alpha = 0.5, label = 'sw_versicolor')
sw_virginica = np.array(dataset.iloc[100:150, 1])
pyplot.hist(sw_virginica, alpha = 0.5, label = 'sw_virginica')
plt.xlabel("Sepal width")
plt.ylabel("No.of Flowers")
plt.legend()
plt.show()

pl_setosa = np.array(dataset.iloc[0:50, 2])
pyplot.hist(pl_setosa, alpha = 0.5, label = 'pl_setosa')
pl_versicolor = np.array(dataset.iloc[50:100, 2])
pyplot.hist(pl_versicolor, alpha = 0.5, label = 'pl_versicolor')
pl_virginica = np.array(dataset.iloc[100:150, 2])
pyplot.hist(pl_virginica, alpha = 0.5, label = 'pl_virginica')
plt.xlabel("Petal Length")
plt.ylabel("No.of Flowers")
plt.legend()
plt.show()

pw_setosa = np.array(dataset.iloc[0:50, 3])
pyplot.hist(pw_setosa, alpha = 0.5, label = 'pw_setosa')
pw_versicolor = np.array(dataset.iloc[50:100, 3])
pyplot.hist(pw_versicolor, alpha = 0.5, label = 'pw_versicolor')
pw_virginica = np.array(dataset.iloc[100:150, 3])
pyplot.hist(pw_virginica, alpha = 0.5, label = 'pw_virginica')
plt.xlabel("Petal width")
plt.ylabel("No.of Flowers")
plt.legend()
plt.show()

plt.scatter(pw_setosa,pl_setosa)
plt.scatter(pw_versicolor,pl_versicolor)
plt.scatter(pw_virginica,pl_virginica)
plt.show()