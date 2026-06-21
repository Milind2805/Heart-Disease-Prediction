import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
dataset=pd.read_csv(r"D:\Heart_Disease_Prediction.csv")
print("Before:",dataset.isnull().sum())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["Heart Disease"]=le.fit_transform(dataset["Heart Disease"])
#sns.scatterplot(x="Cholesterol",y="Heart Disease",data=dataset)
#plt.show()
#print(dataset.shape)
x=dataset.iloc[:,:-1]
y=dataset["Heart Disease"]
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

lr=LogisticRegression(max_iter=1000)
dt=DecisionTreeClassifier()
sv=SVC()
knn=KNeighborsClassifier()

lr.fit(x_train,y_train)
print("LR TRAIN:",lr.score(x_train,y_train)*100)
print("LR TEST:",lr.score(x_test,y_test)*100)

dt.fit(x_train,y_train)
print("DT TRAIN:",dt.score(x_train,y_train)*100)
print("DT TEST:",dt.score(x_test,y_test)*100)

sv.fit(x_train,y_train)
print("SV TRAIN:",sv.score(x_train,y_train)*100)
print("SV TEST:",sv.score(x_test,y_test)*100)

knn.fit(x_train,y_train)
print("KNN TRAIN:",knn.score(x_train,y_train)*100)
print("KNN TEST:",knn.score(x_test,y_test)*100)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=60)
rf.fit(x_train,y_train)
print(rf.score(x_train,y_train)*100)
print(rf.score(x_test,y_test)*100)
from sklearn.model_selection import cross_val_score
print("Cross val score:", cross_val_score(LogisticRegression(max_iter=1000),x,y,cv=5).mean()*100)

from sklearn.metrics import confusion_matrix
y_pred = lr.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()












