import sys
sys.stdin = open('input.txt', 'r')
line = sys.stdin.readline().strip()
line, n = line.split('/')
n = int(n)
line = line.split(';')
l = len(line)
finished = {}
inform = {}
depend = {}
times = {}
stack = []
for t in line:
    t = t.split(':')
    k = t[0]
    if len(t[1]) == 2:
        tmp = []
    else:
        tmp = t[1][1:-1].split(',')
    finished[k] = False
    if k not in inform:
        inform[k] = []
    if k not in depend:
        depend[k] = []
    if len(tmp) == 0:
        stack.append(k)
    for i in tmp:
        depend[k].append(i)
        if i not in inform:
            inform[i] = [k]
        else:
            inform[i].append(k)
    times[k] = int(t[2])
res = 0
proc = []
cnt = 0
stack.sort()
while cnt<l:
    pl = len(proc)
    proc.extend([[times[t], t] for t in stack[:n-pl]])
    proc.sort()
    stack = stack[n-pl:]
    i = 0
    tmp = []
    while i < len(proc) and proc[i][0]==proc[0][0]:
        k = proc[i][1]
        finished[k] = True
        for ik in inform[k]:
            flag = True
            for dk in depend[ik]:
                if finished[dk]==False:
                    flag = False
                    break
            if flag:
                tmp.append(ik)
        i += 1
    tmp.sort()
    stack.extend(tmp)
    cnt += i
    res += proc[0][0]
    j = i
    while j < len(proc):
        proc[j][0] -= proc[0][0]
    proc = proc[i:]
print(res)