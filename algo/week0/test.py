def f(A, n):
    if (n == 0):
        return 0
    else:
        return f(A, n - 1) + A[n-1]
    

def f2(A,n):
    res = 0
    for i in range(n):
        res+=A[i]
    return(res)

print(f([1,2,3,4,5,6,7,8,14],9))
print(f2([1,2,3,4,5,6,7,8,14],9))
