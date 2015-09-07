def factorial(input_num):
    output = 1

    if input_num > 1:
        for i in range(2, input_num + 1):
            output *= i

    return output

def count_even_numbers(input_list):
    count = 0

    for i in input_list:
        if i % 2 == 0:
            count += 1

    return count

def count_odd_numbers(input_list):
    count = 0

    for i in input_list:
        if i % 2 == 1:
            count += 1

    return count

print("5 factorial is: " + str(factorial(5)))

import datetime
now = datetime.datetime.now()
print ("Current date and time : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))

input_numbers = [1,2,3,4,5,6,7,8,9]
print(input_numbers)
print("There are {} even numbers".format(count_even_numbers(input_numbers)))
print("There are {} odd numbers".format(count_odd_numbers(input_numbers)))

input_numbers = [1,2,3,3,3,3,4,5]
print(input_numbers)
print(set(input_numbers))

person_dict = {"name": "alex", "age": 21, "major": "BIT", "courses": ["db3", "ads", "project"]}
print(person_dict)
