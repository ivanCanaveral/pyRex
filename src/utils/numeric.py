import numpy as np

def extractDigits(number):
    if number > -1:
        digits = [0]*5
        i = 5 - 1
        while i >= 0:
            digits[i] = number%10
            number //= 10
            i -= 1
        return digits

def sigmoid(x):
    x = np.array(x)
    return 1/(1+np.exp(-16*(x-0.7)))