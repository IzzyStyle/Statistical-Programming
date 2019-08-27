import statistics

num_list = [0]

intro = """
CPSC-51100, Summer 2019
Name: Israel Nolazco
PROGRAMMING ASSIGNMENT #1
"""

print (intro)

def inputNumber(userEntry):
    while True:
        try:
            userInput = int(input(userEntry))
        except ValueError:
            print("Not an integer! Try again.")
        else:
            return userInput
            break
#The function above serves to make sure we are getting numbers in the input field
#If anything else but an interger is enter it will prompt the user to 'try again'
            
            
entry  = inputNumber("Enter a number: " + "(type -1 to finish)")

count = 1    #count is set at 1 because a)serves for the calculation of the current time mean
             #b)stops the while operation to run infinetely
while ( entry > 0):
    
        ++count #takes each iteration to stop and add the number of times the function is running
        
        mean_before = statistics.mean(num_list) #after first entry it takes the mean
                
        num_list.append(entry) #add 'entry' to the list

        var_before = statistics.variance(num_list) # takes the variance of the list
        
        mean_after = statistics.mean(num_list) #Mean of list after user input      
                
        xn = mean_after + ((mean_after - mean_before)/count) #Current Time Mean
        
        
        print("Mean is ",xn) #simple print
                
        if (len(num_list) > 2): #we want to make sure we are comparing two elements in the list for our calculation
                
                sn = (((len(num_list) - 2)/(len(num_list)-1))* var_before) + ((mean_after - mean_before)**2/len(num_list))
                
                print("Variance is ","{:.1f}".format(sn))
                #formatting the results for one decimal point to make it look like the example.
                #results show slightly different, assuming its because I am using the stats package
            
                entry  = inputNumber("Enter a number: (type -1 to finish)")
                #this allows for the user to keep entering their input and guides them to make sure they can quit properly
        else:
                entry  = inputNumber("Enter a number: (type -1 to finish)")           
                #the code is repeated in this else statement because we created an if scenario for a list with more than two elements

if (count == -1):
    
        print("The program has ended")
else:
        print("Whoops! Try typing -1 if you want to quit, otherwise make sure its a positive integer")
        #In case of accidental negative sign