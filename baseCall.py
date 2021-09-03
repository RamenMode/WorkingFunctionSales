import test as Arceus
import pandas as pd

SalesDataSet = pd.read_pickle('MarylandVehicleSales2002-2021') #load your pkl or use pd.read_csv #1
IndicatorDataSet = pd.read_csv('historical_country_United_States_indicator_Inflation_Rate.csv') # load your pkl for indicator #2
    # Note: Your indicator csv file must be custom edited to include the values for the dates you need and move it up. You can keep all other columns unchanged as you will specify the indexOfIndicator
    # 1954-06-30T00:00:00. Eventually, We can parse the csv file indicator values until the end date is reached. I will work on an algorithm for that
IndexColumnOfIndicator = 3 #which index is your indicator columns #3
SalesColumnName = 'New' # specify your sales column. New is my sales in this case #4
TestYear = 2018 # prediction year #5
NumNodes = 153 # approx 2/3 of the number of your rows. Make this divisible by a three #6

# Note: One optimization that needs to be developed is configuring the excel doc by datetime.
# If anyone can figure this out that would be great. I have searched numerous solutions on stack overflow
# but datatypes dont match. With more time I'll be able to figure this out. 
# Message me on discord if you need help, but I'll demo during our Friday meeting









Arceus.Arceus(SalesDataSet, IndicatorDataSet, IndexColumnOfIndicator, SalesColumnName, TestYear, NumNodes)