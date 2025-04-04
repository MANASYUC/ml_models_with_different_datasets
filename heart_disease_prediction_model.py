import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("heart_disease_data.csv")
data.head()
#target column == heart_disease
data['sex'] = data['sex'].apply(lambda x: 1 if x == 'Male' else 0)
data['sex']
data['diabetes'] = data['diabetes'].apply(lambda x: 1 if x == 'Yes' else 0)
data['diabetes']
data.info()
X = data[['age', 'cholesterol', 'bp', 'sex', 'diabetes']]
y = data['heart_disease']

X_train,  X_test, y_train,y_test = train_test_split(X,y,test_size=.4)
lg = LogisticRegression()
lg.fit(X_train,y_train)
predictions = lg.predict(X_test)
predictions = np.round(predictions)
classification_report(y_test, predictions)
confusion_matrix(y_test, predictions)
# give about 50 percent accuracy, not really good, 
# possibly because of less data, only had 5 features to train on