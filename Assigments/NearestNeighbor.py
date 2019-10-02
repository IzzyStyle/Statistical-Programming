#import NumPy 
import numpy as np
import os
#introduction that will be printed in beginning of the program
intro = """CPSC-51100, Summer 2019
NAME: Israel Nolazco
PROGRAMMING ASSIGNMENT #3"""
print(intro)

#print empty line to make it visibly better
print()
# Opens file in local directory, read each line and creates list for each line
def open_file(filename):
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    location = location + os.sep + filename
    lines = open(location)
    file_list = [(x.rstrip()) for x in lines]
    return file_list

#opens the file and gives that open file a name for reference
training_data = open_file("iris-training-data.csv")


#this will split the subelemts that have a ',' in training_data
training_split = [i.split(',') for i in training_data]

#will create a n empty list
flat_training_data = []

#will create a list of the labels for Training Data
for sublist in training_split: 
    for val in sublist:
        if val == 'Iris-setosa' in sublist:
            flat_training_data.append(val)
        if val == 'Iris-versicolor' in sublist:
            flat_training_data.append(val)
        if val == 'Iris-virginica' in sublist:
            flat_training_data.append(val)

#turns list into an arry.
#this is my 1D array from the training_data(labels)               
training_label  =  np.asarray(flat_training_data)

#2D array from training_data
trainingarr = np.loadtxt(training_data,
delimiter = ',', ndmin=2, usecols =(0,1,2,3))
 
#opens the file and gives that open file a name for reference
testing_data = open_file("iris-testing-data.csv")

#this will split the subelemts that have a ',' in testing data
testing_split = [i.split(',') for i in testing_data]

#will create a n empty list
flat_testing_data = []

#will create a list of the labels for testing Data
for sublist in training_split: 
    for val in sublist:
        if val == 'Iris-setosa' in sublist:
            flat_testing_data.append(val)
        if val == 'Iris-versicolor' in sublist:
            flat_testing_data.append(val)
        if val == 'Iris-virginica' in sublist:
            flat_testing_data.append(val)

#turns list into an arry.
#this is my 1D array from the training_data(labels)               
testing_label  =  np.asarray(flat_testing_data)


#2D array from testing_data       
testingarr = np.loadtxt(testing_data,
delimiter = ',', ndmin=2, usecols = (0,1,2,3))


#taking first row of training arr, but needs to be reshapen since its flat
rainattr_1 = trainingarr[:,0]

#taking first row of testing arr, but needs to be reshapen since its flat
raintest_1 = testingarr[:,0]

#reshaping traning arr to look like a column
reattr_1 = rainattr_1.reshape(-1,1)

#rehspaing testing arr to look like a row
retest_1 = raintest_1.reshape(1,-1)

#gives an array 75 x 75 with distance values subtracted and square
newarr_1 = (reattr_1 - retest_1)**2

#taking second row of training arr, but needs to be reshapen since its flat
rainattr_2 = trainingarr[:,1]

#taking second row of testing arr, but needs to be reshapen since its flat
raintest_2 = testingarr[:,1]

#reshaping traning arr to look like a column
reattr_2 = rainattr_2.reshape(-1,1)

#rehspaing testing arr to look like a row
retest_2 = raintest_2.reshape(1,-1)

#gives an array 75 x 75 with distance values subtracted and square
newarr_2 = (reattr_2 - retest_2)**2

#taking third row of training arr, but needs to be reshapen since its flat
rainattr_3 = trainingarr[:,2]

#taking third row of testing arr, but needs to be reshapen since its flat
raintest_3 = testingarr[:,2]

#reshaping traning arr to look like a column
reattr_3 = rainattr_3.reshape(-1,1)

#rehspaing testing arr to look like a row
retest_3 = raintest_3.reshape(1,-1)

#gives an array 75 x 75 with distance values subtracted and square
newarr_3 = (reattr_3 - retest_3)**2

#taking fourth row of training arr, but needs to be reshapen since its flat
rainattr_4 = trainingarr[:,3]

#taking fourth row of testing arr, but needs to be reshapen since its flat
raintest_4 = testingarr[:,3]

#reshaping traning arr to look like a column
reattr_4 = rainattr_4.reshape(-1,1)

#rehspaing testing arr to look like a row
retest_4 = raintest_4.reshape(1,-1)

#gives an array 75 x 75 with distance values subtracted and square
newarr_4 = (reattr_4 - retest_4)**2

#adds the new arrays creating new 75x75 array and square root results
distance = np.sqrt(newarr_1 + newarr_2 + newarr_3 + newarr_4)

#getting the locations of min values in the 75x75 array
#results collarate with location with elemnt location in testing label
min_distance = np.argmin(distance,axis = 1)

#reshaping testing label to be an array with one dimension
newtest = testing_label.reshape(1,-1)

#reshaping training label to be an array with one dimension
newtraining = training_label.reshape(-1,1)

#for loop creates a print of each number and compares results with
#training and testing labels
for x in range(len(min_distance)):
    n = training_label[min_distance[x]]
    b= testing_label[x]
    a = str(x + 1)
    print("number " + a + " " +n + " " + b )

#empty list for accuracy calculation
accuracy_list = []

#adds element location inside of training label while being compare with min-dis
for x in range(len(min_distance)): 
    accuracy_list.append(training_label[min_distance[x]])

#turns list into array    
accuracy_arr = np.asarray(accuracy_list)

#tells me how many times each element is similar
accuracy_typenp = np.sum(accuracy_arr==testing_label)

#since results from count is numpyfloat needs to be change to regular float
accuracy_int = accuracy_typenp.item()

#with regular float taking the lenght of the list testing_label to get result
#multiply by 100 to get results
accuracy_results = (accuracy_int / len(testing_label))*100

#printing accuracy results
print("Accuracy: " + str(accuracy_results) + "%")