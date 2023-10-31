# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the linear regression using gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Use the standard libraries in python for Gradient Design.
2.Upload the dataset and check any null value using .isnull() function.
3.Declare the default values for linear regression.
4.Calculate the loss usinng Mean Square Error.
5.Predict the value of y.
6.Plot the graph respect to hours and scores using scatter plot function.

## Program:
~~~

Program to implement the linear regression using gradient descent.
Developed by: Manoj M
RegisterNumber: 212221240027

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1 (1).txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):

  m=len(y) # length of the training data
  h=X.dot(theta) # hypothesis
  square_err=(h-y)**2

  return 1/(2*m) *np.sum(square_err) # returning J

data_n= data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error= np.dot(X.transpose(),(predictions-y))
    descent= alpha* 1/m *error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):

  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profict of $"+str(round(predict2,0)))

~~~

## Output:

### Profit Prediction graph


![1](https://github.com/Manoj21500566/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94588708/9ce81e0f-640a-4c7a-b1fa-ab793b3c7e2e)

### Compute Cost Value


![2](https://github.com/Manoj21500566/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94588708/8bb1b3e1-15ad-4c32-b06a-87836134a376)

### h(x) Value


![3](https://github.com/Manoj21500566/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94588708/2c62d6a7-1fb1-42d5-9b49-1879b6332236)

### Cost function using Gradient Descent Graph


![4](https://github.com/Manoj21500566/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94588708/683aa22a-13d4-43fa-b6f9-445d4fe368ed)

### Profit Prediction Graph


![5](https://github.com/Manoj21500566/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94588708/8c3d5d2a-6609-4d17-88de-17a6c755108a)

### Profit for the Population 35,000


![6](https://github.com/Manoj21500566/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94588708/285eb7c6-0929-45e1-8fbf-936c0faddc7b)

### Profit for the Population 70,000


![7](https://github.com/Manoj21500566/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94588708/bf0e0476-ddb9-4fed-862a-cb954147bc4a)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
