import random

def quickSort(array, low=None, high=None):
    if low is None or high is None:
        low = 0
        high = len(array) - 1

    if (low < high):
        pi = partition(array, low, high)

        array=quickSort(array, low, pi - 1)
        array=quickSort(array, pi + 1, high)
    return(array)
    
def partition(array, low, high):
    pivot = array[high]
    
    i = (low - 1)

    for j in range(low,high):
        if (array[j] < pivot):
            i+=1    
            array[i], array[j] = array[j], array[i]
        
    
    array[i + 1], array[high] = array[high], array[i + 1]
    return (i + 1)


A = ([random.randint(1, 9) for _ in range(10)])
print(A)
print(quickSort(A))
