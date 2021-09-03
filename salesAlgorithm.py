import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from sklearn.model_selection import train_test_split

def iChooseYouPikachu(dataframe, testYear, indexOfIndicator, indicatorData, numNodes): # add a couple parameters about input size/layers, etc.
    # dataframe with an index 0 to length - 1, sales as the first column, followed by year, and month
    # not optimized for multiple indicators yet, will continue updating for multiple indicator
    IndicatorValues = indicatorData.iloc[:, indexOfIndicator]
    print(IndicatorValues)
    print(dataframe)
    # print(frameTemp['Year'])
    dataframe['Indicator'] = IndicatorValues
    dataframe.rename(columns = {list(dataframe)[0]: 'Sales'}, inplace = True)
    testData = dataframe.loc[dataframe['Year'] == testYear]
    testData = testData.set_index([pd.Index(list(range(12)))])
    print(testData)
    anotherData = dataframe.set_index('Year')
    trainData = anotherData.drop(testYear, axis = 0)
    trainData = trainData.reset_index()
    trainData = trainData[['Sales', 'Year', 'Month', 'Indicator']] # add 'UnemploymentRateValues'
    print(trainData)
    dates = trainData.index
    #print("dates: ", dates)

    # Now we isolate training and test
    X = trainData.iloc[:, 1:4] # increase index
    Y = trainData.iloc[:, 0]
    Xi_Test = testData.iloc[:, 1:4] # increase index
    Yi_Test = testData.iloc[:, 0]

    # Model Function
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.2, shuffle = True)
    model = Sequential()
    model.add(Dense(numNodes*5/3, activation = 'relu', input_dim=3)) # increase input_dim
    model.add(Dense(numNodes*2/3, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(numNodes*1/3, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')


    model.fit(XTrain, YTrain, epochs=100, batch_size=128)
    #results from model
    loss = model.evaluate(XTest, YTest)
    print('sqrt loss', np.sqrt(loss))
    print('standard deviation', trainData['Sales'].std())

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
    new_predictions_series.to_csv("predicted_sales.csv",header=False)

    print(dataframe)
    print('tested for ')

        
