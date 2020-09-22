def selection():
    arr = [28, 94, 100, 39, 52, 17, 42, 73]
    arr.sort()
    res = []
    for i in range(0, len(arr)):
        res.append(sum(arr[:i+1]))
        # res += tmp
    res = sum(res)
    print(res, res/len(arr))
def monster():
    import sys
    n, y = [int(t) for t in sys.stdin.readline().strip().split()]
    arr = []
    for i in range(n):
        arr.append([
            int(t) for t in sys.stdin.readline().strip().split()
        ])
    arr.sort(key=lambda x: x[0])
    res = 0
    for i in range(n):
        break

def element():
    import sys
    arr = [int(t) for t in sys.stdin.readline().strip().split()]
    # arr = [1, 2, 3, 4]
    s = sum(arr)
    mn = min(arr)
    mx = max(arr)
    x_mx = sum([t-mn for t in arr])
    flag = False
    for i in range(x_mx):
        if i%2 == 1:
            continue
        j = i >> 1
        s_left = s - j
        if s_left%4 != 0:
            continue
        flag = True
        break
    if flag:
        return s_left
    else:
        return -1
print(element())





a, b, c = 0.4236, 0.6496, 0.7514
print(c-a-b)
print(2.5*a+8.5*b-5.5*c)