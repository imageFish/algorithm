def mySum():
    import sys
    sys.stdin = open('input.txt', 'r')
    n = int(sys.stdin.readline().strip())
    arr = [int(t) for t in sys.stdin.readline().strip().split()]

    m = 1000000007
    bt = [0]*(n+1)
    def lowbit(i):
        return i&-i
    def update(i, k):
        while i<=n:
            bt[i] = (k + bt[i]%m)%m
            i += lowbit(i)
    def get_sum(i):
        res = 0
        while i>0:
            res = (res + bt[i])%m
            i -= lowbit(i)
        return res

    for i,k in enumerate(arr):
        update(i+1, k)

    q = int(sys.stdin.readline().strip())
    res = 0
    for i in range(q):
        l, r = [int(t) for t in sys.stdin.readline().strip().split()]
        res += (get_sum(r+1) - get_sum(l))%m
    print(res)

import sys
#sys.stdin = open('input.txt', 'r')
n = int(sys.stdin.readline().strip())
data = [
    [[0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0]],
    [[0, 0], [0, 0]],
    [0, 0]
]
for _ in range(n):
    tmp = [int(t) for t in sys.stdin.readline().strip().split()]
    spam = tmp[-1]
    for i in range(4):
        data[i][tmp[i]][spam] += 1
    data[-1][spam] += 1
m = int(sys.stdin.readline().strip())
res = []
for i in range(m):
    tmp = [int(t) for t in sys.stdin.readline().strip().split()]
    d = []
    for spam in [0, 1]:
        t = 1
        for i in range(4):
            t *= data[i][tmp[i]][spam]/data[-1][spam]
        d.append(t)
    _res = []
    for spam in [0, 1]:
        t = d[spam] * data[-1][spam] / n
        _res.append(t)
    if _res[0] > _res[1]:
        res.append(0)
    else:
        res.append(1)
print(' '.join([str(t) for t in res]))