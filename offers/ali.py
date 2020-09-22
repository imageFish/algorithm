import sys
def same_order():
    wf = open('offers/input2.txt', 'r')
    sys.stdin = wf
    T = int(sys.stdin.readline().strip())
    for _ in range(T):
        n = int(sys.stdin.readline().strip())
        x = [int(t) for t in sys.stdin.readline().strip().split()]
        y = [int(t) for t in sys.stdin.readline().strip().split()]

        arr = [[i,j] for i,j in zip(x,y)]
        arr.sort()
        y_rev_min = [0]*n
        y_rev_min[-1] = arr[-1][1]
        for i in range(n-2, -1, -1):
            y_rev_min[i] = min(y_rev_min[i+1], arr[i][1])
        _x = _y = -1
        res = 0
        for i in range(n):
            if y_rev_min[i]==arr[i][1] and arr[i][0]>_x and arr[i][1]>_y:
                res += 1
                _x, _y = arr[i]
        print(res)
def hit_balls():
    wf = open('offers/input2.txt', 'r')
    sys.stdin = wf
    arr 
same_order()