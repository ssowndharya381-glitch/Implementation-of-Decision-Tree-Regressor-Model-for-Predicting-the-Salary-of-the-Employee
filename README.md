# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the salary dataset.  
2. Split the dataset into training data and testing data.  
3. Create and train the Decision Tree Regressor model using the training data.  
4. Predict the salary, evaluate the model performance, and display the decision tree   

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SOWNDHARYA S
Register Number: 212225220100
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("Salary.csv")
print(df.head())

X = df[["Level"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

model = DecisionTreeRegressor(
    max_depth=4,
    random_state=1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE :", mean_absolute_error(y_test, y_pred))

mse = mean_squared_error(y_test, y_pred)
print("MSE :", mse)
print("RMSE :", np.sqrt(mse))
print("R2 Score :", r2_score(y_test, y_pred))

plt.figure(figsize=(15,8))

plot_tree(
    model,
    feature_names=["Level"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree Regressor")
plt.show()

new_level = [[5]]
salary = model.predict(new_level)
print("Predicted Salary:", salary[0])
```

## Output:
<img width="521" height="377" alt="image" src="https://github.com/user-attachments/assets/302cfc0f-7755-455d-bca7-61d1bcbea610" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
