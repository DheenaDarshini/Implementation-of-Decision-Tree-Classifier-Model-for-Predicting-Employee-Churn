# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Data – Import the employee dataset with relevant features and churn labels.

2. Preprocess Data – Handle missing values, encode categorical features, and split into train/test sets.

3. Initialize Model – Create a DecisionTreeClassifier with desired parameters.

4. Train Model – Fit the model on the training data.

5. Evaluate Model – Predict on test data and check accuracy, precision, recall, etc.

6. Visualize & Interpret – Visualize the tree and identify key features influencing churn.
 
## Program and Output:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Dheena Darshini Karthik Dheepan
RegisterNumber:  212223240030
*/
```
```
import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()
```
![alt text](<Screenshot 2025-04-24 090536.png>)
```
data.info()
data.isnull().sum()
data['left'].value_counts()
```
![alt text](<Screenshot 2025-04-24 090546.png>)
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
```
![alt text](<Screenshot 2025-04-24 090559.png>)
```
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
```
![alt text](<Screenshot 2025-04-24 090611.png>)
```
y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
```
![alt text](<Screenshot 2025-04-24 090623.png>)
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![alt text](<Screenshot 2025-04-24 090632.png>)
```
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree # Import the plot_tree function

plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
![alt text](<Screenshot 2025-04-24 090641.png>)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
