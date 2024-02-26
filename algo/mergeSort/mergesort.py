import random

def mergesort(A,startIndex):
    if len(A) <=1:
        return A
    
    split=len(A)//2
    l,r = mergesort(A[:split],startIndex), mergesort(A[split:],startIndex+split)

    il = 0
    ir = 0

    mergedArray = []

    while il<len(l) and ir<len(r):
        if(l[il]<r[ir]):
            #print(Array[startIndex+il]<Array[startIndex+split+ir])
            mergedArray.append(l[il])
            il +=1
        else:
            mergedArray.append(r[ir])
            ir +=1

    mergedArray+=(r[ir:]+l[il:])
    
    return mergedArray
    
Array = ([random.randint(1, 9) for _ in range(10)])
ms = (mergesort(Array,0))
print("sorted array is \a",ms)
print(sorted(Array) == ms)
