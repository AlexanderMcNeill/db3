import math

class Test:
    def __init__(self):
        self.value = ""

    def getString(self):
        input("Please input a word: ")

    def printString(self):
        print(upper(self.value))

class Circle():

    def __init__(self, radius):
        self.radius = radius

    def area():
        return math.pi * self.radius ^ 2

class Person():

    def __init__(self, age):
        self.age = age

    def get_gender():
        pass

class Male(Person):

    def __init__(self, age):
        super(Male, self).__init__()
        self.arg = arg

    def get_gender():
        return "Male"

class Female(Person):

    def __init__(self, age):
        super(Female, self).__init__()

    def get_gender():
        return "Female"

class AlexException(Exception):
    """Exception thrown when Alex has a meltdown"""
    def __init__(self, message):
        super(AlexException, self).__init__(message)


raise RuntimeError()
