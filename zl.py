M=[]
class DFS_hungary():
 
    def __init__(self, nx, ny, edge, cx, cy, visited):
        self.nx, self.ny=nx, ny
        self.edge = edge
        self.cx, self.cy=cx,cy
        self.visited=visited
 
    def max_match(self):
        res=0
        for i in self.nx:
            if self.cx[i]==-1:
                for key in self.ny:
                    self.visited[key]=0
                res+=self.path(i)
        return res
 
    def path(self, u):
        for v in self.ny:
            if self.edge[u][v] and (not self.visited[v]):
                self.visited[v]=1
                if self.cy[v]==-1:
                    self.cx[u] = v
                    self.cy[v] = u
                    M.append((u,v))
                    return 1
                else:
                    M.remove((self.cy[v], v))
                    if self.path(self.cy[v]):
                        self.cx[u] = v
                        self.cy[v] = u
                        M.append((u, v))
                        return 1
        return 0
def _data():
    import sys
    boys = [int(i) for i in sys.stdin.readline().strip().split()]
    girls = [int(i) for i in sys.stdin.readline().strip().split()]
    n = int(sys.stdin.readline().strip())
    edges = {}
    cb, cg = {b:-1 for b in boys}, {g:-1 for g in girls}
    visited = {g:0 for g in girls}
    for b in boys:
        edges[b] = {}
        for g in girls:
            edges[b][g] = 0
    for _ in range(n):
        b, g = [int(i) for i in sys.stdin.readline().strip().split()]
        edges[b][g] = 1
    print(DFS_hungary(boys, girls, edges, cb, cg, visited).max_match())

def cal(n, m):
    def _cal(n1, n2, opt):
        if opt=='+': return n1+n2
        if opt=='-': return n1-n2
        if opt=='*': return n1*n2
        if opt=='/': return n1/n2
        assert False
    opts = ['+', '-', '*', '/']
    for l in opts:
        for j in opts:
            for k in opts:
                equ = [n, l, n, j, n, k, n]
                while len(equ)>1:
                    for i in range(len(equ)):
                        if equ[i] in ['*', '/']:
                            break
                    if i < len(equ)-1:
                        t = _cal(equ[i-1], equ[i+1], equ[i])
                        equ = equ[:i-1] + [t] + equ[i+2:]
                    else:
                        t = _cal(equ[0], equ[2], equ[1])
                        equ = [t] + equ[3:]
                if equ[0] == m: return 1
    return 0


def maze():
    import sys
    sys.stdin = open('input.txt', 'r')
    t = int(sys.stdin.readline().strip())
    for _ in range(t):
        n, m = [int(t) for t in sys.stdin.readline().strip().split()]
        costs = []
        used = []
        arr = []
        si = sj = None
        for _ in range(n):
            costs.append([-1]*m)
            used.append([False]*m)
            arr.append(sys.stdin.readline().strip())
            if not si:
                for i in range(m):
                    if arr[-1][i] == '@':
                        si, sj = len(arr)-1, i
                        break
        que1 = [[si, sj]]
        costs[si][sj] = 0
        used[si][sj] = True

        def dfs(i, j, c):
            costs[i][j] = c
            res = []
            for _i, _j in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
                _i += i
                _j += j
                if _i<0 or _i>=n or _j<0 or _j>=m:
                    return False
                elif used[_i][_j]:
                    continue
                used[_i][_j] = True
                if arr[_i][_j]=='.':
                    tmp = dfs(_i, _j, c) 
                    if tmp == False:
                        return False
                    res.extend(tmp)
                elif arr[_i][_j]=='*':
                    res.append([_i, _j])
            return res
            
        res = 0
        flag = False
        while que1:
            que2 = []
            while que1:
                i, j = que1.pop()
                tmp = dfs(i, j, res)
                if tmp == False:
                    flag = True
                    print(res)
                    break
                que2.extend(tmp)
            if flag: break
            res += 1
            que1 = que2
        if flag: continue
        print(-1)
maze()