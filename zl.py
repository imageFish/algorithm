def password():
    a = [i for i in range(10)]
    b = [chr(i) for i in range(ord('a'), ord('z')+1)]
    a += b
    al = len(a)

    res = []
    for i in range(al): 
        for j in range(al):
            for k in range(al):
                if i != j and i != k and j != k:
                    res.append([a[i], a[j], a[k]])
    return res
password()