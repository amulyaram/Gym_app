import numpy as np
import pandas as pd
df=pd.read_excel(r'c:\ML\dataGym.xlsx')
del df['BMI']
del df['Class']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Prediction']=le.fit_transform(df['Prediction'])
#split into x and y
x=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=30)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=30)
model.fit(x_train,y_train)
predicted=model.predict(x_test)
#print(predicted)
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
print(accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted))
print(confusion_matrix(y_test, predicted))
import pickle
pickle.dump(model, open("Model_GYM.pkl", "wb"))
model1=pickle.load(open("Model_GYM.pkl", "rb"))
print(model1.predict([[40,5.6,70]]))

