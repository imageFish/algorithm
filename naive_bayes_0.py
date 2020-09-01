import sys
input_stream = open('input.txt', 'r')
sys.stdin = input_stream
M, N, d = [int(t) for t in input().split()]
Y = [0, 0]
X = {}
w_cnt = 0
for i in range(M):
    line = [int(t) for t in input().split()]
    label = line[0]
    w_cnt += len(line) - 1
    Y[label] += 1
    for w in enumerate(line[1:]):
        if w in X:
            X[w][label] += 1
        else:
            tmp = [0, 0]
            tmp[label] = 1
            X[w] = tmp
val = []
val_X = set()
for i in range(N):
    line = [int(t) for t in input().split() if t != '?']
    val.append(line)
    for w in line:
        if w in X or w in val_X:
            continue
        else:
            val_X.add(w)
for line in val:
    y = [1, 1]
    yl = len(y)
    sigma = 2
    y = [y[t] * (Y[t]+1) / (M+sigma) for t in range(yl)]
    sigma = 2
    for w in line:
        if w in X:
            sample_xj = [t for t in X[w]]
        else:
            sample_xj = [0 for _ in range(yl)]
        w_cnt = sum(sample_xj)
        xj_p = [(t+1)/(w_cnt+sigma) for t in sample_xj]
        y = [y[t]*xj_p[t] for t in range(yl)]
    #     print(xj_p)
    #     print(y)
    # print()
    if y[0] > y[1]:
        print(0)
    else:
        print(1)