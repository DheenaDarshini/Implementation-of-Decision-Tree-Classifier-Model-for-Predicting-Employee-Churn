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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv(r"E:\Desktop\CSE\Introduction To Machine Learning\dataset\Employee.csv")
print(data.head())
print(data.info())
data.isnull().sum()
print(data["left"].value_counts())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
print(x.head())
y=data["left"]
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
pre=dt.predict([[0.5,0.8,9,260,6,0,1,2]])
print(pre)

plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```
## Output:

![11](https://github.com/user-attachments/assets/c4ebad5f-2fbf-4f1e-acc3-8d515ce790c8)

![22](https://github.com/user-attachments/assets/8fb7a9a9-0443-405e-8964-502a8cc1f05e)

![33](https://github.com/user-attachments/assets/dc99868e-3dad-4a65-afee-b69b72510461)

![44](https://github.com/user-attachments/assets/88447cc2-e24b-4026-afa2-3793f250a579)

![55](https://github.com/user-attachments/assets/e2ddf4ca-5129-4b53-836d-0af96bcdd141)

![66](https://github.com/user-attachments/assets/7126860c-ec80-46ab-9b0b-d7f74193e17e)

![77](https://github.com/user-attachments/assets/bacf79cf-98b3-42e9-81a1-14ac4b8f9e3c)

![88](https://github.com/user-attachments/assets/30267649-e18d-4bdc-b45c-9b1f8e569c4c)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
