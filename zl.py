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