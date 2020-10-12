def test1():
    import sys
    from functools import cmp_to_key
    def cmp_fun(x, y):
        if x[0]==y[0]:
            return x[1] - y[1]
        return y[0] - x[0]
    arr = [
        [5, 4], [5, 2], [8, 7], [8, 5], [3, 2], [12, 0]
    ]
    arr1 = sorted(arr, key=cmp_to_key(cmp_fun))
    print(arr1)
def test2():
    import sys
    sys.stdin = open('input.txt', 'r')
    # m, n = int(input()), int(input())
    m, n = [int(t) for t in sys.stdin.readline().strip().split()]
    arr = [[0]*(n+1)]
    for _ in range(m):
        tmp = [0]
        for c in sys.stdin.readline().strip():
            if c=='M':
                tmp.append(1)
            else:
                tmp.append(0)
        arr.append(tmp)
    res = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            if arr[i][j]==1:
                arr[i][j] = min(arr[i-1][j], arr[i][j-1], arr[i-1][j-1]) + 1
                res = max(res, arr[i][j])
    res *= res
    print(res)
def test3():
    import sys
    sys.stdin = open('input.txt', 'r')
    n, m = [int(t) for t in sys.stdin.readline().strip().split()]
    if n==1:
        print(1, 1)
        return
    in_cnt, out_cnt = [0]*(n+1), [0]*(n+1)
    for _ in range(m):
        i, j = [int(t) for t in sys.stdin.readline().strip().split()]
        in_cnt[i] += 1
        out_cnt[j] += 1
    in0, in0out0, out0 = 0, 0, 0
    mn = mx = 1
    for i in range(1, n+1):
        if in_cnt[i]==0 and out_cnt[i]==0:
            in0out0 += 1
        if in_cnt[i]==0:
            mn = i
            in0 += 1
        if out_cnt[i]==0:
            mx = i
            out0 += 1
    if in0out0>0 or in0!=1:
        mn = -1
    if in0out0>0 or out0!=1:
        mx = -1
    print(mx, mn)
test2()