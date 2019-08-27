from pandas import Series, DataFrame
import pandas as pd
import numpy as np

intro = """CPSC-51100, Summer 2019
NAME: Israel Nolazco
PROGRAMMING ASSIGNMENT #5"""
print(intro)

#print empty line to make intro visibly better
print()

#creates reference for file
cps_data = pd.read_csv('cps.csv')

#create dataframe with only specific columns
cpsdf = DataFrame(cps_data,columns=['School_ID','Short_Name','Is_High_School','Zip','Student_Count_Total','College_Enrollment_Rate_School','Grades_Offered_All','School_Hours'])

#splits column Grades_Offered_All and uses first column for lowest value
Lowest_Grade_Offered = cpsdf['Grades_Offered_All'].str.split(",").str[0]

#splits column Grades_Offered_All and uses last column for highest value
Highest_Grade_Offered = cpsdf['Grades_Offered_All'].str.split(",").str[-1]

#split column School_Hours and uses the starting time
starting_time = cpsdf['School_Hours'].str.split("-").str[0].fillna('00:00 AM')

#split column to just get hour value
Starting_Hour = starting_time.replace({'7:':7,'8:':8,'9:':9},regex=True)

#calculating standard deviation of College_Enrollment_Rate_School
college_std = cpsdf.loc[:,"College_Enrollment_Rate_School"].std()

#calculating mean of College_Enrollment_Rate_School
college_mean = cpsdf.loc[:,"College_Enrollment_Rate_School"].mean()

#replacing missing values with mean
college_col = DataFrame(cps_data,columns=["College_Enrollment_Rate_School"]).fillna(college_mean)

#creates dataframe to be viewed
batch = DataFrame(cps_data,columns=['School_ID','Short_Name','Is_High_School','Zip','Student_Count_Total'])

#dataframe that will be printed
view_batch = pd.concat([batch,college_col,Lowest_Grade_Offered,Highest_Grade_Offered,Starting_Hour],axis = 1)

#getting dataframe for nonhighschools
non_highschols = DataFrame(cps_data,columns=['Is_High_School', 'Student_Count_Total'])

#deleting rows that contain True for Is_High_School
non_highschols_total = non_highschols[non_highschols['Is_High_School'] == False]

#calculating mean of Student_Count_Total
view_non_meanHS = non_highschols_total.loc[:,'Student_Count_Total'].mean()

#calculating standard deviation of Student_Count_Total
view_non_stdHS = non_highschols_total.loc[:,'Student_Count_Total'].std()

#number of schools in zipcode
hs_inzip = cpsdf.loc[:,"Zip"].isin([60601, 60602, 60603, 60604, 60605, 60606, 60607,60616])

#Number of schools outside of zip
remove_inzip = hs_inzip[hs_inzip==False].count()

#Getting distribution times
eight_time = Starting_Hour.isin([8])
seven_time = Starting_Hour.isin([7])
nine_time = Starting_Hour.isin([9])

#giving generic value to print easily 
a = eight_time[eight_time==True].count()
b = seven_time[seven_time==True].count()
c = nine_time[nine_time==True].count()

print(view_batch.head(10))
print ("College Enrollment Rate for High Schools =" + "{:.2f}".format(college_mean) + " " + "(std =" +"{:.2f}".format(college_std) + ")" )
print()
print ("Total Student Count for non-High Schools =" + "{:.2f}".format(view_non_meanHS) + " " + "(std =" + "{:.2f}".format(view_non_stdHS) + ")" )
print()
print ("Number of schools outside of the loop: " + str(remove_inzip))
print()
print ("Distribution of Starting Hours\n" + "8am: " +str(a) + "\n7am: " + str(b) + "\n9am: "+ str(c))
