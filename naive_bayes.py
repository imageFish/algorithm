import sys
input_stream = open('input.txt', 'r')
sys.stdin = input_stream
M, N, d = [int(t) for t in input().split()]
X = [{} for _ in range(d+1)]
for i in range(M):
    line = [int(t) for t in input().split()]
    label = line[0]
    for j, xj in enumerate(line):
        if xj in X[j]:
            X[j][xj][label] += 1
        else:
            tmp = [0 for _ in range(2)]
            tmp[label] += 1
            X[j][xj] = tmp
for i in range(N):
    line = [int(t) if t!='?' else t for t in input().split()]
    y = [1, 1]
    for j, xj in enumerate(line):
        if j == 0:
            xj_sigma = 2
            sample_xj = [X[0][0][0], X[0][1][1]]
        else:
            if xj in X[j]:
                sample_xj = [t for t in X[j][xj]]
            else:
                sample_xj = [0 for _ in range(len(y))]
            xj_sigma = 2
        xj_d = len(X[j])
        
        xj_p = [(t+1)/(xj_d+xj_sigma) for t in sample_xj]
        y = [y[t]*xj_p[t] for t in range(len(y))]
        print(xj_p)
        print(y)
    print()
    if y[0] > y[1]:
        print(0)
    else:
        print(1)