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
print(cal(1, 4))
print(cal(1, 3))