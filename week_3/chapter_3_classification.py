from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt


mnist = fetch_openml('mnist_784', version = 1, data_home="~/Documents/UNSW/2022H5 Data mining and Machine Learning/week_3/")

mnist.keys()

X,y = mnist['data'], mnist['target']

X.shape
y.shape

X.head()

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap = 'binary')
plt.axis("off")
plt.show()