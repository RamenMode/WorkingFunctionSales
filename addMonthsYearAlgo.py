import pandas as pd
def IChooseYouBulbasaur(dataframe, startYear, endYear, startMonth, endMonth):
    # startYear is the starting year of the data (ex. 2009)
    #endYear is the end year of the data (ex. 2021)
    # startMonth is the index of the month (1 is january, 12 is december)
    # endMonth is the same as above
    # add a dataframe with an incrementing index and sales
    # all variables are numbers (1 is January, etc.)
    columnMonth = []
    columnYear = []
    columnIndicator = []
    for p in range(startMonth, 13):
        columnMonth.append(p)
        columnYear.append(startYear)
        columnIndicator.append(0)
    for x in range(startYear + 1, endYear):
        for i in range(1, 13):
            columnMonth.append(i)
            columnYear.append(x)
            columnIndicator.append(0)
    for r in range(1, endMonth + 1):
        columnMonth.append(r)
        columnYear.append(endYear)
        columnIndicator.append(0)
    dataframe['Year'] = columnYear
    dataframe['Month'] = columnMonth
    dataframe['Indicator'] = columnIndicator
    return dataframe






