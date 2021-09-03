import pandas as pd
import numpy as np
import tensorflow as tf
import addMonthsYearAlgo as Bulbasaur
import salesAlgorithm as Pikachu
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from sklearn.model_selection import train_test_split

def Arceus(data, indicatorData, indexOfIndicator, salesColumn, testYear, numNodes):
    dataNew = pd.DataFrame(data)
    dataNew = dataNew[[salesColumn]] # specify your sales column. New is my sales in this case #5
    dataNew = Bulbasaur.IChooseYouBulbasaur(dataNew, 2002, 2021, 1, 4) # specify start year, end year, start month, end month
    #the data now has 3 columns: Sales, Month, Year, indicator (all 0s won't make a differnce)
    print(dataNew)
    Pikachu.iChooseYouPikachu(dataNew, testYear, indexOfIndicator, indicatorData, numNodes)


# Algorithm to automatically get the dates you need for indicator values - not complete
# Specify how many input layers you need - completed
# optimize clarity of the code - completed
# can add multiple indicators - Not complete Will add once we have several successful singleton indicators to work with. What seems to work, what doesn't
# can produce multiple indicators for the whole thing (suggested by Nick) - future update