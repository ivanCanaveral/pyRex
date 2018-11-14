from time import time

def epoch_s():
    return int(time())

def epoch_ms():
    return int(time()*1000)