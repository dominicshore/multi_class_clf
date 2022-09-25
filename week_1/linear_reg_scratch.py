import os
import pandas as pd
from  numpy import *
import matplotlib.pyplot as plt


df = genfromtxt("Week 1/data_linearreg.csv", delimiter=",")

plt.scatter(df[0], df[1])
plt.show()


df[0, 1]