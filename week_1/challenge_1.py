from tkinter.ttk import Style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("./week_1/iris_data/iris.csv", header=0, index_col=[0])
iris.columns = map(str.lower, iris.columns)

# Use simple find and replace to convert the class labels to 1, 2, and 3 in the dataset. 
# iris['species'] = np.where(
#     iris['species'].equals('setosa'), 
#     int(1), 
#     np.where(
#         iris['species'].equals("versicolor"), int(2), int(4)
#     )
# )

iris.loc[iris['species']=="setosa", 'species'] = int(1)
iris.loc[iris['species']=="versicolor", 'species'] = int(2)
iris.loc[iris['species']=="virginica", 'species'] = int(3)

# Read the data and report mean and standard deviation for each column in the features (4 features)
iris.describe()

# Report the class distribution (i. e number of instances for each class)
iris['species'].value_counts()

# Show histogram for each feature. We recommend you to use a single function/method that outputs the histogram with a given filename. eg. feature1.png which is given as a parameter to the function. A for loop should be used to call the function/method
plt.hist(iris['sepal.length'])
plt.show()
plt.close()

# pd.wide_to_long(iris, ['sepal.', 'petal.'], i = "species", j = "type")
def graphs_n_stuff(dataset):
    graph_iris = pd.melt(dataset, id_vars='species', value_vars=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
    sns.set_theme(style="darkgrid")
    sns.displot(graph_iris, x = "value", col = "species", row = "variable")
    plt.show()
    plt.close()

graphs_n_stuff(iris)

# Split data into a train and test. Use 60 percent data in the training and test set which is assigned i. randomly ii. assigned by the first 60 percent as train and the rest as test. 
train_rand = iris.sample(frac=0.6)
test_rand = iris.sample(frac=0.4)

train_non_rand = iris[iris.index<=np.percentile(iris.index, 60)]
test_non_rand = iris[iris.index>np.percentile(iris.index, 60)]

# Use previous functions to report the mean and standard deviation of the train and test set and class distribution and also the histograms for each feature. 
train_rand.describe()
test_rand.describe()

graphs_n_stuff(train_rand)
graphs_n_stuff(test_rand)

# Create another subset of the train and test set where only 1 feature selected by the user makes the dataset with the class. This means that you create a dataset with any 1 of the 4 features. 

# Create a subset of the dataset where you consider only instances that feature class 1 or 2, so that you treat this problem as a binary classification problem later, i.e save it as binary_iristrain.txt and binary_iristest.txt. Carry out the stats and visuals in Step 6 for this dataset. 

# Can you normalise the input features between [0 and 1]? Write code that can do so and save normalised versions.