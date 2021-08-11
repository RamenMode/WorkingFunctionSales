import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from sklearn.model_selection import train_test_split
data = pd.read_pickle('MarylandVehicleSales2002-2021')
dataNew = pd.DataFrame(data)
print(dataNew)
dataNew = dataNew.drop(columns = ['Total Sales New', 'Total Sales Used'])
print(dataNew)


column = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]
dataNew['Month'] = column
print(dataNew)

for i in range(0, 232):
    dataNew.iloc[i, 2] = dataNew.iloc[i, 2] + dataNew.iloc[i, 3]
dataNew = dataNew.drop(columns = ['Used', 'Month '])
dataNew = dataNew.rename(columns={'New': 'Total Sales'})
print(dataNew)

dataNew = dataNew[['Total Sales', 'Year ', 'Month']]
print(dataNew)
#What are we testing with?
testData = dataNew.loc[dataNew['Year '] == 2019]
testData = testData.set_index([pd.Index(list(range(12)))])
print(testData)
anotherData = dataNew.set_index('Year ')
trainData = anotherData.drop(2019, axis = 0)
trainData = trainData.reset_index()
trainData = trainData[['Total Sales', 'Year ', 'Month']]
print(trainData)
dates = trainData.index
#print("dates: ", dates)

# Now we isolate training and test
X = trainData.iloc[:, 1:3]
Y = trainData.iloc[:, 0]
Xi_Test = testData.iloc[:, 1:3]
Yi_Test = testData.iloc[:, 0]

# Model Function
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.2, shuffle = True)
model = Sequential()
model.add(Dense(450, activation = 'relu', input_dim=2))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'linear'))

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(XTrain, YTrain, epochs=100, batch_size=128)

#results from model
loss = model.evaluate(XTest, YTest)
print('sqrt loss', np.sqrt(loss))
print('standard deviation', trainData['Total Sales'].std())

predictions = model.predict(X)

predictions_list = map(lambda x: x[0], predictions)
print('predlist', predictions_list)
predictions_series = pd.Series(predictions_list,index=dates)
dates_series = pd.Series(dates)

Predicted_sales = model.predict(Xi_Test)
new_dates_series=pd.Series(Xi_Test.index)
new_predictions_list = map(lambda x: x[0], Predicted_sales)
new_predictions_series = pd.Series(new_predictions_list,index=new_dates_series)

#export to csv
new_predictions_series.to_csv("predicted_saless.csv",header=False)

print(dataNew)
# Author: Maryland Gov
# https://catalog.data.gov/dataset?publisher=opendata.maryland.gov&organization=state-of-maryland
# Used and New Car Data sales monthly
# https://catalog.data.gov/dataset/mva-vehicle-sales-counts-by-month-for-calendar-year-2002-2020-up-to-october
