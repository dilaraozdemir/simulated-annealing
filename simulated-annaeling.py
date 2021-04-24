# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:09:52 2021

@author: rhaen
"""

import pandas as pd
import random
import math
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv("covtype.csv")

#%%
df = df[df["Cover_Type"] < 3]
a = df.iloc[:2000,:-1]
y = df.iloc[:2000,-1:]

# Changing column names of x dataframe with 0 and 1
c = np.random.random(54) > 0.5
c = c.astype(int).astype(str)

a.columns = c


# Taking only 1's of feature for knn
x = a.loc[:,["1"]]


#%% train test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)


#%% KNN
knn = KNeighborsClassifier(n_neighbors=5)
#%% K fold Cross Validation

accuracies = cross_val_score(estimator=knn,X = x_train, y = y_train.values.ravel(), cv = 5)
accuraciess = np.mean(accuracies)
print("average accuracy: ",np.mean(accuracies))
print("average std: ",np.std(accuracies))



#%%
def get_neighbors(a):
    c = np.random.random(54) > 0.5
    c = c.astype(int).astype(str)
    a.columns = c
    a = a.xs('1', axis=1)
    return a
    
# neigbor parameter is created by get_neighbor so it has randomly selected 1's
# y is target to predicited
def get_cost(x,y):
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
    knn = KNeighborsClassifier(n_neighbors=3)
    accuracies = cross_val_score(estimator=knn,X = x_train, y = y_train.values.ravel(), cv = 5)
    return np.mean(accuracies)

#%% SA
# param = dataframe a with 0's and 1's, initial_state is knn score with df a (only 1's selected)
def simulated_annealing(param, initial_state, y):
    """Peforms simulated annealing to find a solution"""
    
    
    initial_temp = 90
    final_temp = .1
    alpha = 0.01
    
    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = initial_state
    solution = current_state

    while current_temp >=  final_temp:
        neighbor = get_neighbors(param)
        # Check if neighbor is best so far
        cost_diff = get_cost(neighbor,y)

        if cost_diff < math.exp(cost_diff / current_temp):
            solution = cost_diff
        # decrement the temperature
        current_temp -= alpha
    return solution

#%%
print("score", simulated_annealing(a, accuraciess, y))



