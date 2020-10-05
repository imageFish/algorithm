def demo1():
    import sys, math
    sys.stdin = open('input.txt', 'r')
    x = int(sys.stdin.readline().strip())
    d0 = d1 = 1 # gcd(d0, d1) = 1
    start = 3
    while True:
        p = 1
        while (x-d0*p)%d1!=0 and d0*p<x:
            p += 1
        if d0*p >= x:
            break
        print('{} {}'.format(start, math.ceil((x-d0*p)/(d0*d1)))) # gcd(d0, d1)=1
        t = d0
        d0 = d1
        d1 = t + d0
        start += 1
def demo2():
    import sys
    sys.stdin = open('input.txt', 'r')
    netrons = list(input())
    k = int(input())
    false_set = {}
    def eliminate(netrons, st):
        l = r = st
        while l>=0 and r<len(netrons):
            _l, _r = l, r
            while l>=0 and netrons[l]==netrons[st]:
                l -= 1
            while r<len(netrons) and netrons[r]==netrons[st]:
                r += 1
            if not (l<_l and _r<r and _l-l+r-_r>=3):
                return netrons[:_l+1] + netrons[_r:]
            # l, r = _l, _r
            st = r
        return netrons[:l+1] + netrons[r:]
    def eliminate_sort(netrons, cost):
        res = []
        i = 0
        while i < len(netrons):
            j = i
            while j<len(netrons) and netrons[j]==netrons[i]:
                j += 1
            flag = False
            if j-i >= 2 and 1 <= cost:
                tmp = netrons[:i] + [netrons[i]] + netrons[i:]
                flag = True
            elif j-i == 1 and 2 <= cost:
                tmp = netrons[:i] + [netrons[i], netrons[i]] + netrons[i:]
            else:
                i = j
                continue
            _netrons = eliminate(tmp, i)
            if flag:
                _res = [[netrons[i], i]]
                _cost = cost - 1
            else:
                _res = [[netrons[i], i], [netrons[i], i]]
                _cost = cost - 2
            res.append([_netrons, _res, _cost])
            i = j
        return res

            
    def bfs(netrons, cost):
        if not netrons and cost>=0:
            return True, []
        if netrons and cost<=0:
            return False, []
        es = eliminate_sort(netrons, cost)
        es.sort(key=lambda x: len(x[0]))
        for e in es:
            k = ''.join(e[0])
            if k in false_set and false_set[k]>= e[2]:
                continue
            _flag, _res = bfs(e[0], e[2])
            if _flag:
                e[1].extend(_res)
                return True, e[1]
            else:
                if k in false_set:
                    false_set[k] = max(false_set[k], e[2])
                else:
                    false_set[k] = e[2]
        return False, []
    flag, _res = bfs(netrons, k)
    for t in _res:
        print('{} {}'.format(*t))
demo2()
        
            
            


    
class Solution:
    def findMinStep(self, board: str, cost:int) -> int:
        def elimination(s, start):
            if start < 0:
                return s
            # 从start开始对s进行消去
            len_s = len(s)
            # if len_s <= 2&nbs***bsp;start==len_s-1:
            #     return s
            if s[start+1]!=s[start]:
                return s
            l = start
            r = start
            while l > 0 and s[l - 1] == s[start]:
                l -= 1
 
            while r < len_s - 1 and s[r + 1] == s[start]:
                r += 1
            if r - l <= 1:
                return s
            else:
                temp = s[:l] + s[r + 1:]
                return elimination(temp, l - 1)
 
        def dfs(board,cost,step):
            if (not board) and cost>=0:
                return True,step
            if cost<=0:
                return False,[]
            i =0
 
            while i<len(board):
                res = cost
                restep = step[:]
                j = i+1
                while j<len(board) and board[i]==board[j]:
                    j+=1
                if j-i==1:
                    restep+=[board[i]+' '+str(i),board[i]+' '+str(i)]
                    res -= 2
                    temp = elimination(board[:i]+board[i]+board[i]+board[i:],i)
                else:
                    restep+=[board[i]+' '+str(i)]
                    res -= 1
                    temp = elimination(board[:i] + board[i] + board[i:], i)
                flag,restep = dfs(temp,res,restep)
                if flag:
                    return True,restep
                else:
                    i = j
            return False,[]
 
        flag,step = dfs(board,cost,[])
        return step if flag else []
import sys
sys.stdin = open('input.txt', 'r')
board = input()
cost = int(input())
k = Solution().findMinStep(board,cost)
for i in k:
    print(i)

def demo3():
    import sys, collections
    sys.setrecursionlimit(350000)
    sys.stdin = open('input.txt', 'r')
    tree = collections.defaultdict(list)
    k, n = [int(t) for t in sys.stdin.readline().strip().split()]
    for _ in range(n):
        tmp = [int(t) for t in sys.stdin.readline().strip().split()]
        assert tmp[1] not in tree
        if len(tmp) > 2:
            tree[tmp[1]] = tmp[2:]
    def dfs(p):
        if len(tree[p])==0:
            return 0
        dp = []
        for c in tree[p]:
            dp.append(dfs(c))
        dp.sort(reverse=True)
        i = 0
        tl = len(tree[p])
        n = tl // k
        if tl%k != 0:
            n += 1
        res = -1
        while i < n:
            res = max(res, max(dp[i*k:i*k+k])+i+1)
            i += 1
        return res
    res = dfs(0)
    return res
demo1()