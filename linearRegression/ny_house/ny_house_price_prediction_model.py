import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


df = pd.read_csv("NY-House-Dataset.csv")
#target column == PRICE
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

y = new_df['PRICE']
X = new_df.drop('PRICE',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3)

model = LinearRegression()

model.fit(X_train, y_train)

predict = model.predict(X_test)
mae = metrics.mean_absolute_error(y_test, predict)
mse = metrics.mean_squared_error(y_test, predict)
rmse = np.sqrt(mse)

# Print evaluation metrics
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
