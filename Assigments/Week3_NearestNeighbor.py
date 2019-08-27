from pandas import Series, DataFrame
import pandas as pd
import numpy as np

intro = """CPSC-51100, Summer 2019
NAME: Israel Nolazco
PROGRAMMING ASSIGNMENT #4"""
print(intro)

#print empty line to make intro visibly better
print()

#opens cars file       
cars_data = open("cars.csv")

#reads csv data and gives it a reference name.
cars_data = pd.read_csv(cars_data)

#using data and creates a DataFrame
carsdf = DataFrame(cars_data, columns = ['make','aspiration'])

#unpivots dataframe for long format for better handleing
cars_newframe = carsdf.melt()

#computes cross tabulation for each factor
#creates two rows for aspiration
#creates a column for each make
cars = pd.crosstab(index=df['aspiration'], columns=df['make'])

#total number of features
#creates a series
totalfeat = cars.sum()

#broadcasting to calculate probablity
con_prob =  (cars/totalfeat)*100

#gives an ndarray of the values in the new dataframe
prob_val = con_prob.values

#creates a flat array from the prob_value ndarray
prob_arr = prob_val.flatten('F')

#creates list of make values in cars data frame
#repeat list to make it 44 values, to match prob_arr size
make_list = list(cars.columns)*2


#creates list of aspiration value in cars data frame 
#repeats list to make 44 values, to match prob_arr size   
asp_list = list(cars.index)*22

#print each item in the flat array
#print each item in make list print array
#print each item in asp list print array
for x in range(len(prob_arr)):
    a = str(round(prob_arr[x],2))
    b = asp_list[x]
    c = make_list[x]
    print("Prob(aspiration="+b + "|make ="+ c + "= " + a + "%")
    
#creates series of a count value of make column
#Also sorts series alphabetically by the index
make_s = carsdf['make'].value_counts().sort_index()

#adds the count of values in make_s series
total_make = make_s.sum()

#car probability in datafram
car_prob = (make_s/total_make)*100

#prints extra space for better visual 
print()

#prints probility of make over the entire series
for index, value in car_prob.iteritems():
    val = str(round(value,2))
    print ( "Prob(make= "+ index + ")"+ "= " + val + "%")