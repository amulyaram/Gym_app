#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[97]:


df=pd.read_excel('dataGYM.xlsx')


# In[98]:


df.head()


# In[99]:


#del df['BMI']


# In[100]:


del df['Prediction']


# In[101]:


df.head()


# In[102]:


df.info()


# In[103]:


df['Class'].unique()


# In[104]:


df["Class"].replace({"EXtremely obese":"Extremely obese", "Healthy\xa0":"Healthy", "Under weight":"Underweight"}, inplace=True)
df["Class"].unique()


# In[105]:


df.columns = ['Age', 'Height(feet)', 'weight(pounds)', 'BMI', 'Class']


# In[106]:


df["weight(pounds)"] = df["weight(pounds)"]*2.2


# In[107]:


df.head()


# In[108]:


df["Age"].hist()


# In[109]:


df["Age"].hist()


# In[110]:


import seaborn as sns


# In[111]:


sns.countplot(df["Class"])


# In[112]:


x = df.iloc[:,:-2]
y = df.iloc[:,-1]


# In[113]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[114]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=30, stratify=y)
for i in [x_train, x_test, y_train, y_test]:
  print(i.shape)


# In[115]:


from sklearn.model_selection import GridSearchCV
parameters = {'criterion':('gini', 'entropy'), 'max_depth':[8, 10,12]}
dt = DecisionTreeClassifier()
clf = GridSearchCV(dt, parameters)
clf.fit(x_train, y_train)
clf.best_params_


# In[116]:


y_pred1= clf.predict(x_test)
y_pred1


# In[ ]:





# In[117]:


def accuracy(model):
  y_pred_train = model.predict(x_train)
  print(f"Train accuracy: {accuracy_score(y_train, y_pred_train)}")
  y_pred_test = model.predict(x_test)
  print(f"Test accuracy: {accuracy_score(y_test, y_pred_test)}")
  print(f"\nClassification Report(test_data):\n {classification_report(y_test, y_pred_test)}")
  print(f"Confustion_metrix(test_data):\n\n {confusion_matrix(y_test, y_pred_test)}")


# In[118]:


accuracy(clf)


# In[119]:


sv_classifier = SVC(kernel="linear")


# In[120]:


sv_classifier.fit(x_train, y_train)


# In[121]:


accuracy(sv_classifier)


# In[122]:


Logistic=LogisticRegression()


# In[123]:


Logistic.fit(x_train, y_train)


# In[124]:


accuracy(Logistic)


# In[125]:


Random=RandomForestClassifier(n_estimators=20)


# In[126]:


Random.fit(x_train,y_train)


# In[127]:


accuracy(Random)


# In[128]:


def gym_app():
  age = input("Your Age: ")
  height = input("Your height in feet: ")
  weight = input("Your weight in pounds: ")
  input_data = np.asarray([age, height, weight])
  input_data = input_data.reshape(1,-1)
  prediction  = clf.predict(input_data)
  prediction2 = Random.predict(input_data)
  print(f"Decision tree Classifier: {prediction[0]}\nRandom_forest Classifier: {prediction2[0]}")


# In[131]:


gym_app()


# In[ ]:


#both decision tree and random forests are giving good accuracy but for deploying random forests are god. so i ma using random forests to pickle model


# In[133]:


import pickle
pickle.dump(Random, open("Random_model.pkl", "wb"))


# In[ ]:




