
from sklearn import svm
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#Jack did part 1a, cole did part 1b, walker did part 2

testPath = '/Users/walkerbasham/Desktop/DataScience/DataScienceHW2/HW2/Lab2/part1/test.csv'
trainPath = '/Users/walkerbasham/Desktop/DataScience/DataScienceHW2/HW2/Lab2/part1/train.csv'
pt2Train = '/Users/walkerbasham/Desktop/DataScience/DataScienceHW2/HW2/Lab2/part2/part2Train.csv'
pt2Test = '/Users/walkerbasham/Desktop/DataScience/DataScienceHW2/HW2/Lab2/part2/part2Test.csv'

print("PART 1a")

test = pd.read_csv(testPath,) # load data
train = pd.read_csv(trainPath,) # load data

#Answer to quesiton 1
#use a combination of pandas library and sklearn to produce confusion matrix with the decision tree classifier
print("Answer to question 1")
print()
X_train = train.drop(['id', 'target'], axis=1)
X_test = test.drop(['id', 'target'], axis=1)
y_train = train['target']
y_test = test['target']

DTClassifier = DecisionTreeClassifier(splitter = 'best', max_depth = 15)


DTClassifier.fit(X_train,y_train)

test_pred = DTClassifier.predict(X_test) # prediction of test data
train_pred = DTClassifier.predict(X_train) # prediction of train data

test_cm = confusion_matrix(y_test, test_pred)
train_cm = confusion_matrix(y_train, train_pred)


print("This is the confusion matrix for train.csv using a decision tree model")
print(train_cm)
print()
print("This is the confusion matrix for test.csv, using a Decision Tree Model")
print(test_cm)


#Answer to quesiton 2
#use the precision_recall_fscore_support function of sklearn to generate the precision, recall and fscore for train.csv

print()
print()
print("Answer to question 2.  The format of the answer is (precision, recall, fscore) of train.csv using DecisionTree")
print(precision_recall_fscore_support(y_train, train_pred, average='binary'))

#Answer to quesiton 3
#use the precision_recall_fscore_support function of sklearn to generate the precision, recall and fscore for test.csv

print()
print()
print("Answer to question 3.  The format of the answer is (precision, recall, fscore) of test.csv using DecisionTree")
print(precision_recall_fscore_support(y_test, test_pred, average='binary'))

#Answer to quesiton 4
#comparing precision, recall and f1score between train.csv and test.csv
print()
print("Answer to number 4: ")
print("As you can see, the precison, recall and f1 score are significantly higher on train.csv as opposed to train.csv. These changes happen because test.csv is much more imbalanced than train.csv  ")
print()

#Answer to quesiton 5
#use the confusion matricies and the .ravel function to compute tpr and fpr of train.csv

tn, fp, fn, tp = train_cm.ravel()
tn1, fp1, fn1, tp1 = test_cm.ravel()
print("answer to number 5. This is th TPR and FPR of the train data")
TPR = tp/(tp+fn)
FPR = fp/(fp+tn)
print("TPR of train.csv: " + str(TPR))
print("FPR of train.csv: " + str(FPR))
print()

#Answer to quesiton 6
#use the confusion matricies and the .ravel function to compute tpr and fpr of test.csv

print("answer to number 6. This is th TPR and FPR of the test data")
TPR1 = tp1/(tp1+fn1)
FPR1 = fp1/(fp1+tn1)
print("TPR of test.csv: " + str(TPR1))
print("FPR of test.csv: " + str(FPR1))


#Answer to quesiton 7
#compare TPR and FPR of test.csv and train.csv
print()
print()
print("Answer to number 7: ")
print("TPR measures the proportion of actual positives that are correctly identified as such. FPR is calculated as the ratio between the number of negative events wrongly categorized as positive (false positives) and the total number of actual negative events (regardless of classification). As you can see from the data, train.csv has a much higher true positive rate than test.csv. This means that train.csv is classifying positives at a hihger rate than test.csv. In addition, the FPR of train is much lower than the FPR of test, meaning it classifies positives falsely at a much lower rate than when the same classifier is run on test.csv.  ")

#Answer to quesiton 8
#what is best to judge quality of the model, precision and recall or tpr and fpr?
print()
print("Answer to number 8: ")
print("To start with, recall is the same as TPR, so we are really only looking at the distance between FPR and precision. The main difference between these two types of metrics is that precision denominator contains the False positives while false positive rate denominator contains the true negatives. While precision measures the probability of a sample classified as positive to actually be positive, the false positive rate measures the ratio of false positives within the negative samples. So with a dataset with a large number of negative samples, precision is likely better because Precision is not affected by a large number of negative samples. This is because precision measures the number of true positives out of the samples predicted as positives (TP+FP).")

##################part1b###########################################
print()
print()
print()

test = pd.read_csv(testPath,) # load data
train = pd.read_csv(trainPath,) # load data

#Answer to quesiton 1
#use a combination of pandas library and sklearn to produce confusion matrix with the random forest classifier
print("Answer to question 1")
print()
X_train = train.drop(['id', 'target'], axis=1)
X_test = test.drop(['id', 'target'], axis=1)
y_train = train['target']
y_test = test['target']

RFClassifier = RandomForestClassifier(n_estimators=500, random_state=0)

RFClassifier.fit(X_train,y_train)

test_pred = RFClassifier.predict(X_test) # prediction of test data
train_pred = RFClassifier.predict(X_train) # prediction of train data

test_cm = confusion_matrix(y_test, test_pred)#create confusion matrix of test data
train_cm = confusion_matrix(y_train, train_pred)#create confusion matrix of train data


print("This is the confusion matrix for train.csv using a Random Forest model")
print(train_cm)
print()
print("This is the confusion matrix for test.csv, using a Random Forest Model")
print(test_cm)


#Answer to quesiton 2
#use the precision_recall_fscore_support function of sklearn to generate the precision, recall and fscore for train.csv
print()
print()
print("Answer to question 2.  The format of the answer is (precision, recall, fscore) of train.csv using DecisionTree")
print(precision_recall_fscore_support(y_train, train_pred, average='binary'))

#Answer to quesiton 3
#use the precision_recall_fscore_support function of sklearn to generate the precision, recall and fscore for test.csv
print()
print()
print("Answer to question 3.  The format of the answer is (precision, recall, fscore) of test.csv using DecisionTree")
print(precision_recall_fscore_support(y_test, test_pred, average='binary'))


#Answer to quesiton 4
#comparing precision, recall and f1score between train.csv and test.csv
print()
print("Answer to number 4: ")
print("As you can see, the precison, recall and f1 score are 1 or almost 1 on train.csv as opposed to train.csv. These changes happen because test.csv is much more imbalanced than train.csv, and every single instance was classified correctly on train.csv  ")
print()

#Answer to quesiton 5
#use the confusion matricies and the .ravel function to compute tpr and fpr of train.csv
tn, fp, fn, tp = train_cm.ravel()
tn1, fp1, fn1, tp1 = test_cm.ravel()
print("answer to number 5. This is th TPR and FPR of the train data")
TPR = tp/(tp+fn)
FPR = fp/(fp+tn)
print("TPR of train.csv: " + str(TPR))
print("FPR of train.csv: " + str(FPR))

#Answer to quesiton 6
#use the confusion matricies and the .ravel function to compute tpr and fpr of train.csv
print()
print("answer to number 6. This is th TPR and FPR of the test data")
TPR1 = tp1/(tp1+fn1)
FPR1 = fp1/(fp1+tn1)
print("TPR of test.csv: " + str(TPR1))
print("FPR of test.csv: " + str(FPR1))

#Answer to quesiton 7
#compare TPR and FPR of test.csv and train.csv
print()
print()
print("Answer to number 7: ")
print("TPR measures the proportion of actual positives that are correctly identified as such. FPR is calculated as the ratio between the number of negative events wrongly categorized as positive (false positives) and the total number of actual negative events (regardless of classification). As you can see from the data, the TPR and FPR are 1.0 or almost 1.0. This means that every instance in the dataset was classified correctly. The same cannot be said about train.csv however.")

print()
print()
###############part2#############

print("PART 2")
print()
i = 4
#Answer to quesiton 1
#use a combination of pandas library and sklearn to produce confusion matrix with decision tree, svm, logistic regression, and artificial neural network
trainDT = pd.read_csv('/Users/walkerbasham/Desktop/DataScience/DataScienceHW2/HW2/Lab2/part2/part2Train.csv',) # load data
TrainX = trainDT.drop(trainDT.columns[i], axis=1)
TrainY = trainDT[trainDT.columns[i]]

testDT = pd.read_csv('/Users/walkerbasham/Desktop/DataScience/DataScienceHW2/HW2/Lab2/part2/part2Test.csv',) # load data
TestX = testDT.drop(testDT.columns[i], axis=1)
TestY = testDT[testDT.columns[i]]

classifierDecisionTree = DecisionTreeClassifier(splitter = 'best', max_depth = 15)
classifierDecisionTree.fit(TrainX, TrainY)
y_predDecisionTree = classifierDecisionTree.predict(TestX)


print("This is the confusion matrix classified with decisionTree, using the part2Train.csv as the training data and part2Test.csv as the testing data")
print(confusion_matrix(TestY, y_predDecisionTree))

print()
#print("the format of the answer is (precision, recall, fscore")
#print(precision_recall_fscore_support(TestY, y_predDecisionTree, average='binary'))

#################SVM
trainSVM = pd.read_csv('/Users/walkerbasham/Desktop/DataScience/DataScienceHW2/HW2/Lab2/part2/part2Train.csv',) # load data
TrainXSVM = trainSVM.drop(trainSVM.columns[i], axis=1)
TrainYSVM = trainSVM[trainSVM.columns[i]]

testSVM = pd.read_csv('/Users/walkerbasham/Desktop/DataScience/DataScienceHW2/HW2/Lab2/part2/part2Test.csv',) # load data
TestXSVM = testSVM.drop(testSVM.columns[i], axis=1)
TestYSVM = testSVM[testSVM.columns[i]]

classifierSVM = svm.SVC(gamma='scale')
classifierSVM.fit(TrainXSVM,TrainYSVM)
y_predSVMClassifier = classifierSVM.predict(TestXSVM)
print("This is the confusion matrix classified with SVM, using the part2Train.csv as the training data and part2Test.csv as the testing data")

print(confusion_matrix(TestYSVM, y_predSVMClassifier))

#print("the format of the answer is (precision, recall, fscore")
#print(precision_recall_fscore_support(TestYSVM, y_predSVMClassifier, average='binary'))

#################LogisticRegression
trainLR = pd.read_csv(pt2Train,) # load data
TrainXLR = trainLR.drop(trainLR.columns[i], axis=1)
TrainYLR = trainLR[trainLR.columns[i]]

testLR = pd.read_csv(pt2Test,) # load data
TestXLR = testLR.drop(testLR.columns[i], axis=1)
TestYLR = testLR[testLR.columns[i]]


classifierLR = LogisticRegression(random_state=0, solver='saga', max_iter = 500, multi_class='multinomial').fit(TrainXLR, TrainYLR)
y_predLRClassifier = classifierLR.predict(TestXLR)
print("This is the confusion matrix classified with Logistic Regression, using the part2Train.csv as the training data and part2Test.csv as the testing data")
print(confusion_matrix(TestYLR, y_predLRClassifier))


print()
#print("the format of the answer is (precision, recall, fscore")
#print(precision_recall_fscore_support(TestYLR, y_predLRClassifier, average='binary'))

#################ArtificialNeuralNetwork
trainANN = pd.read_csv('/Users/walkerbasham/Desktop/DataScience/DataScienceHW2/HW2/Lab2/part2/part2Train.csv',) # load data
i = 4
TrainXANN = trainANN.drop(trainANN.columns[i], axis=1)
TrainYANN = trainANN[trainANN.columns[i]]

testANN = pd.read_csv('/Users/walkerbasham/Desktop/DataScience/DataScienceHW2/HW2/Lab2/part2/part2Test.csv',) # load data
TestXANN = testANN.drop(testANN.columns[i], axis=1)
TestYANN = testANN[testANN.columns[i]]


classifierANN = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, ), random_state=1)
classifierANN.fit(TrainXANN,TrainYANN)

y_predANNClassifier = classifierANN.predict(TestXANN)
print("This is the confusion matrix classified with an Artificial Neural Network, using the part2Train.csv as the training data and part2Test.csv as the testing data")
print(confusion_matrix(TestYANN, y_predANNClassifier))

#Answer to quesiton 2
#use the precision_recall_fscore_support function of sklearn to generate the precision, recall and fscore for the different models
print()
print("Answer to question 2")
print()
print("the format of the answer is (precision, recall, fscore). This is for the Decision Tree Model")
print(precision_recall_fscore_support(TestY, y_predDecisionTree, average='binary'))
print()
print("the format of the answer is (precision, recall, fscore). This is for the SVM Model")
print(precision_recall_fscore_support(TestYSVM, y_predSVMClassifier, average='binary'))
print()
print("the format of the answer is (precision, recall, fscore). This is for the Linear Regression Model")
print(precision_recall_fscore_support(TestYLR, y_predLRClassifier, average='binary'))
print()
print("the format of the answer is (precision, recall, fscore). This is for the Artificial Neural Network Model")
print(precision_recall_fscore_support(TestYANN, y_predANNClassifier, average='binary'))

#Answer to quesiton 2
#comparing the models
print()
print("Answer to question 3")
print()
print("Looking at the data, it appears the SVM model has the highest precision AND f-score, the Artificial Neural Network model has the highest recall. I would say the SVM model is the best model to use as it is the most accurate, however Artificial Neural Network and Decision tree are close. Linear regression has the lowest precision, recall and fscore so I would say it is the worst model to use.")
