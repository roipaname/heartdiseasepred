import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
'exec(% matplotlib inline)'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
data=pd.read_csv("framingham.csv")
data.drop(['education'],inplace=True,axis=1)
data.rename(columns={'male':'Sex_male'},inplace=True)
data.dropna(axis=0,inplace=True)
#print(data['TenYearCHD'].value_counts())
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
X=preprocessing.StandardScaler().fit(X).transform(X)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)
# plt.figure(figsize=(7, 5))
# sns.countplot(x='TenYearCHD', data=data ,palette="BuGn_r")
# plt.show()
# laste=y.plot()
# plt.show()
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
pred=logreg.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy of the model is :",accuracy_score(y_test,pred)*100)
from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test,pred)
cf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))
sns.heatmap(cf_matrix,annot=True,fmt='d',cmap="Greens")
plt.show()
print('The details for confusion matrix is =')
print (classification_report(y_test, pred))
print(pred)
print("==========+++=====")
print(y_test)