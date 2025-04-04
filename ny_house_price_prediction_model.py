import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression

df = pd.read_csv("NY-House-Dataset.csv")
df.info()
df.head()
df.columns

#getting types of houses
df['TYPE'][4796].split()[0]
df['TYPE'] = df['TYPE'].apply(lambda x: x.split()[0])
df['TYPE']


df.drop(columns='ADDRESS', inplace=True)
df.info()

df['STATE'][0].split()[-1]
df['ZIPCODE'] = df['STATE'].apply(lambda x: x.split()[-1])
df.drop(columns='STATE', inplace=True)


df['ADMINISTRATIVE_AREA_LEVEL_2'].value_counts()
df['SUBLOCALITY'].value_counts()
df.drop('SUBLOCALITY', inplace=True, axis=1)


df['MAIN_ADDRESS'].value_counts()
df.drop('MAIN_ADDRESS', axis=1, inplace=True)
df['STREET_NAME'].value_counts()


df['LONG_NAME'].value_counts()
df.drop('LONG_NAME', axis=1, inplace=True)


df['FORMATTED_ADDRESS']
df['test'] = df['FORMATTED_ADDRESS'].apply(lambda x: x.split()[-4].replace(',',''))
df['test'].value_counts()
df[df['test'] == 'York'] == 'New York'
df['test'] = df['test'].apply(lambda x: 'New York' if x == 'York' else x)
df['Neighborhoods and Boroughs'] = df['test']
df.drop('test', inplace=True, axis=1)


df['PROPERTYSQFT'].value_counts()
sns.scatterplot(x='LATITUDE', y='LONGITUDE', data=df, hue='PROPERTYSQFT')
plt.show()

df.drop('ADMINISTRATIVE_AREA_LEVEL_2', axis=1, inplace=True)
df['LOCALITY'].value_counts()
df.drop('FORMATTED_ADDRESS',axis=1,inplace=True)
df.drop('BROKERTITLE', axis=1, inplace=True)
df['TYPE'].value_counts()

#one-hot encoding all values that are not numerical
new_df = pd.get_dummies(df, columns=['LOCALITY', 'STREET_NAME', 'Neighborhoods and Boroughs', 'TYPE'], dtype=int)
len(new_df.columns)

