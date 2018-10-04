import pandas as pd
import random
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Answer to question 1:
# There are 58 explanatory features in the training data.
# 'id' would not be considered an explanatory feature because it does not tell us anything about the data
# it just identifies it.



print("Answer to Question 1")
print('There are 58 explanatory features in the training data. id would not be considered an explanatory feature because it does not tell us anything about the data. It just identifies it')

data = pd.read_csv("train.csv")
# print(data)

# Question 2 starts here
# Answer to question 2 :

print()
print()
print ()
print("Answer to question 2")

columnNames = data.columns.get_values()
binOrCatColumnNames = []
# print(len(columnNames))
countOfbinOrCatColumnNames = 0
current = 0

while current < len(columnNames):
    # print("in while loop")
    if "cat" in columnNames[current] or "bin" in columnNames[current]:
        binOrCatColumnNames.append(columnNames[current])
        countOfbinOrCatColumnNames += 1
    current += 1

# print(binOrCatColumnNames)
print("There are ", countOfbinOrCatColumnNames, "nominal features")

current = 0
while current < countOfbinOrCatColumnNames:
    # print(binOrCatColumnNames[current])
    print("Name of column: ", data[binOrCatColumnNames[current]].value_counts().to_frame())
    current += 1

#question 3 starts here

print ()
print ()
print ()

print("Answer to Question 3")
print("For Question 3, we followed the table in the book. If the data is nominal, we compare the two values. if the data is the same, s = 1. if it is different s = 0. If the data is ordinal, then s = 1 - ((x-y)/n-1)). At the end we averaged these similarities. For this question we used the first two rows. ")

similarityList = []
current = 0
columnValue = 0
rowValue = 0

while current < (len(columnNames)-1):
    x = data.ix[0, columnNames[current]]
    y = data.ix[1, columnNames[current]]
    #print("x is ", x, "y is ", y)
    current += 1
    if "cat" in columnNames[current] or "bin" in columnNames[current]:
        if x == y:
            similarityList.append(1)
            #print("appending 1")

        else:
            similarityList.append(0)
            #print("appending 0")


    else:
        similarityList.append((1-(abs((x-y))/((x+y)-1))))
        #print("appending the functions")
    #print(similarityList)
    current += 1

print("similarity is: ", sum(similarityList)/len(similarityList))


#question 4 starts here
print ()
print ()
print ()
print('Answer to question 4')

totalRows = data.shape[0]

missingDataList = []

numWithMissingData = 0
current = 0
totalNumNegativeOnes = 0
while current < len(columnNames):

    numNegativeOnes = len(data[data[columnNames[current]] == -1])
    totalNumNegativeOnes += numNegativeOnes
    if numNegativeOnes > 0:
        print ("In column ", columnNames[current], " there are ", numNegativeOnes, "instances of -1. This gives us a ratio of: ", float(numNegativeOnes/totalRows))
        missingDataList.append(columnNames[current])
        numWithMissingData += 1


    current += 1

print("There are ", numWithMissingData, "columns with missing data for a total of ", totalNumNegativeOnes, "missing values")

# Question 5
print ()
print ()
print ()
print('Answer to question 5')
print("For question 5, we simply calculated the mean of each column using the .mode() function that can be applied to a column of a datframe column to find the mode.")
print("After caclulating the mode, we filled every value with a -1 in that column with the mode of that particular column. We accomplished this using the .replace function that can be applied to a dataframe column in panda. If the mode was -1, the missing data was filled with the a random value between 0 and the maximum value of the column.")
print()

current = 0
rowCount = 0
negCount = 0

totalNumNegativeOnes = 0
while current < len(columnNames) and negCount < len(missingDataList):
    numNegativeOnes = len(data[data[columnNames[current]] == -1])
    #print("in while loop of question 5")
    columnMode = 0
    columnMode = data[columnNames[current]].mode()
    newColumnMode = columnMode[0]
    #print()
    #print("column name is: ", columnNames[current]," column mode is: ", newColumnMode)
    if columnNames[current] == missingDataList[negCount]:
        if  newColumnMode != -1:
            #print()

            #print("Found negative values in ", columnNames[current], "which matches the missing data list column: ",missingDataList[negCount])
            data[columnNames[current]].replace([-1], [newColumnMode], inplace=True)
            negCount += 1
            #print("Column mode was not -1. new column mode is: ", newColumnMode)
            #print()


        else:

            #print("the mode was -1")
            #print("Found negative values in ", columnNames[current], "which matches the missing data list column: ",missingDataList[negCount])
            negCount += 1
            max = data[columnNames[current]].idxmax()
            #print(max)


            data[columnNames[current]].replace([-1], [random.uniform(0,data.iloc[max, current])], inplace=True)
            #print("Column mode was not -1. replaced with a random int between 0 and the max of this column: ", newColumnMode)



    current += 1

numWithMissingData = 0
current = 0
totalNumNegativeOnes = 0
while current < len(columnNames):

    numNegativeOnes = len(data[data[columnNames[current]] == -1])
    totalNumNegativeOnes += numNegativeOnes
    if numNegativeOnes > 0:
        #print ("In column ", columnNames[current], " there are ", numNegativeOnes, "instances of -1. This gives us a ratio of: ", float(numNegativeOnes/totalRows))
        missingDataList.append(columnNames[current])
        numWithMissingData += 1


    current += 1

print("There are ", numWithMissingData, "columns with missing data for a total of ", totalNumNegativeOnes, "missing values")


# Question 6
print ()
print ()
print ()
print('Answer to question 6')
print("there are two classes, 1 and 0. As you can see from us printing the values of the column, there are 573,518 0s and 21,694 1s.")
print(data['target'].value_counts().to_frame())
print("This is a ratio of 21,694/573,518, or .0378261. This indicates our dataset is highly unbalanced. This is an issue because machine learning algorithms tend to produce unsatisfactory classifiers when faced with imbalanced datasets. For any imbalanced data set, if the event to be predicted belongs to the minority class and the event rate is less than 5%, it is usually referred to as a rare event.")


#question 7
print ()
print ()
print ()
print('Answer to question 7')
print ()
pca = PCA(n_components=10)
principalComponents = pca.fit(data)
data_pca = pca.transform(data)
print(pca.components_)
print("original shape:   ", data.shape)
print("transformed shape:   ", data_pca.shape)

