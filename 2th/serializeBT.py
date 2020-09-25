# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.arr = []
        
    def addNum(self, num: int) -> None:
        import bisect
        bisect.insort(self.arr, num)
    def findMedian(self) -> float:
        nl = len(self.arr)
        if nl%2 == 1:
            return self.arr[(nl+1)//2 - 1 ]
        return (self.arr[nl//2-1]+self.arr[nl//2])/2

class Codec:
    @classmethod
    def serialize(self, root):
        """Encodes a tree to a single string.
        :type root: TreeNode
        :rtype: str
        """
        res = [root.val] if root else []
        pars = [root]
        while True:
            flag = False
            tmp = []
            for n in pars:
                if n!=None:
                    if n.left!=None or n.right!=None: flag=True
                    tmp.extend([n.left, n.right])
            if flag:
                pars = tmp
                res.extend([t.val if t!=None else t for t in tmp])
            else:
                break
        return res
        
    @classmethod
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return None
        root = TreeNode(data[0])
        pars = [root]
        i = 1
        while i < len(data):
            tmp = []
            for n in pars:
                if n!=None:
                    if i<len(data) and data[i]!=None:
                        n.left = TreeNode(data[i])
                    i += 1
                    if i<len(data) and data[i]!=None:
                        n.right = TreeNode(data[i])
                    i += 1
                    tmp.extend([n.left, n.right])
            pars = tmp
        return root

class Solution:
    @classmethod
    def isMatch(self, s, p):
        sl = len(s)
        pl = len(p)
        def dfs(si, pi):
            if si==sl and pi==pl: 
                return True
            if pi==pl and si<sl: 
                return False
            if pi+1<pl and p[pi+1]=='*':
                if dfs(si, pi+2):
                    return True
                while si < sl and (p[pi]=='.' or p[pi]==s[si]):
                    si += 1
                    if dfs(si, pi+2):
                        return True
                if dfs(si, pi+2):
                    return True
                return False
            else:
                if si >= sl: return False
                if p[pi] == '.' or p[pi]==s[si]:
                    return dfs(si+1, pi+1)
                else:
                    return False
        def dp_fun():
            dp = []
            for i in range(sl+1):
                dp.append([False]*(pl+1))
            for i in range(0, sl+1):
                for j in range(0, pl+1):
                    if j == 0:
                        if i==0:
                            dp[i][j] = True
                    else:
                        if p[j-1] == '*':
                            if j>1:
                                dp[i][j] |= dp[i][j-2]
                            if i>0 and (p[j-2]=='.' or p[j-2]==s[i-1]):
                                dp[i][j] |= dp[i-1][j]
                        else:
                            dp[i][j] = i>0 and (p[j-1]=='.' or p[j-1]==s[i-1]) and dp[i-1][j-1]
            return dp[-1][-1]
        return dp_fun()

    @classmethod
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        in_set = {t:i for i,t in enumerate(inorder)}
        def dfs(pl, pr, il, ir):
            if pr < pl: return None
            root = TreeNode(preorder[pl])
            if pl == pr:
                return root
            in_root = in_set[root.val]
            ll = in_root - il
            root.left = dfs(pl+1, pl+ll, il, in_root-1)
            root.right = dfs(pl+ll+1, pr, in_root+1, ir)
            return root
        return dfs(0, len(preorder)-1, 0, len(inorder)-1)

    @classmethod
    def movingCount(self, m, n, k):
        used = []
        for _ in range(m):
            used.append([False]*n)
        def get_sum(k):
            res = 0
            while k>0:
                res += k%10
                k //= 10
            return res
        sum_set = {i:get_sum(i) for i in range(max(m, n))}
        def dfs(i, j):
            res = 0
            for _i, _j in [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]:
                _i += i
                _j += j
                if _i<0 or _i>=m or _j<0 or _j>=n or used[_i][_j]:
                    continue
                used[_i][_j] = True
                if sum_set[_i]+sum_set[_j] > k:
                    continue
                res += 1
                if i!=_i or _j!=j:
                    res += dfs(_i, _j)
            return res
        res = dfs(0, 0)
        return res
    
    @classmethod
    def exist(self, board, word):
        m, n = len(board), len(board[0])
        used = []
        for _ in range(m):
            used.append([False]*n)
        def dfs(i, j, k):
            if board[i][j] != word[k]:
                return False
            used[i][j] = True # 可以使用
            if k == len(word)-1:
                return True
            for _i, _j in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                _i += i
                _j += j
                if _i<0 or _i>=m or _j<0 or _j>=n or used[_i][_j]:
                    continue
                if dfs(_i, _j, k+1): 
                    return True
            used[i][j] = False # 不能使用， 恢复used标记
            return False
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        return False

    @classmethod
    def cuttingRope(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 快速幂运算，利用指数的二进制表示
        k = 1000000007
        def lowbit(k):
            return k&-k
        def mod(p):
            s = {}
            v = 3
            key = 1
            while key <= p:
                s[key] = v
                key <<= 1
                v = v**2
                v %= k
            res = 1 # 指数为0的特例
            while p>0:
                lb = lowbit(p)
                res = (res*s[lb]) % k
                p -= lb
            return res 
        
        if n <= 3:
            return n-1
        p, q = n//3, n%3
        if q==0:
            m = mod(p)
            return m
        if q==1:
            m = mod(p-1)
            return (m*4)%k
        if q==2:
            m = mod(p)
            return (m*q)%k
    
    @classmethod
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        negative = False
        if n<0:
            negative = True
            n = -n
        res = 1
        while n>0:
            if n&1: res *= x
            x = x * x
            n >>= 1
        if negative:
            res = 1/res
        return res
    
    @classmethod
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = s.strip()
        def isInt(x, unsigned=False):
            s = 0
            if s>= len(x): return False
            if x[0] in ['+', '-']:
                if unsigned:
                    return False
                s += 1
            if s>= len(x): return False
            while s<len(x):
                if not ('0'<= x[s] and x[s]<='9'):
                    return False
                s += 1
            return True
        def isDec(x):
            p = None
            for i,c in enumerate(x):
                if c=='.':
                    p = i
                    break
            if p==None: return False
            ll = p
            rl = len(x)-1 - p
            if x[0]=='+' or x[0]=='-':
                ll -= 1
            if ll==0 and rl==0:
                return False
            if ll>0:
                if not isInt(x[:p]):
                    return False
            if rl>0:
                if not isInt(x[p+1:], unsigned=True):
                    return False
            return True
        def isExp(x):
            p = None
            for i,c in enumerate(x):
                if c=='E' or c=='e':
                    p = i
                    break
            if p==None:
                return False
            return (isDec(x[:p]) or isInt(x[:p])) and isInt(x[p+1:])
        return isExp(s) or isDec(s) or isInt(s)

    @classmethod
    def isSubStructure(self, A, B):
        """
        :type A: TreeNode
        :type B: TreeNode
        :rtype: bool
        """
        def same(t1, t2):
            # t1 != None
            if t2==None:
                return False
            if t1.val!=t2.val:
                return False
            if t1.left and not same(t1.left, t2.left):
                return False
            if t1.right and not same(t1.right, t2.right):
                return False
            return True
        if A==None or B==None:
            return False
        pars = [A]
        while pars:
            tmp = []
            for p in pars:
                if same(B, p): return True
                if p.left: tmp.append(p.left)
                if p.right: tmp.append(p.right)
            pars = tmp
        return False
    
    @classmethod
    def longestValidParentheses(self, s):
        res = 0
        stack = []
        sl = len(s)
        dp = [0]*(sl+1)
        for i, c in enumerate(s):
            if c=='(':
                stack.append(i)
            elif c==')':
                if stack:
                    l = stack.pop()
                    dp[i+1] = i-l+1+dp[l]
                    res = max(res, dp[i+1])
        return res

    @classmethod
    def validateStackSequences(self, pushed, popped):
        """
        :type pushed: List[int]
        :type popped: List[int]
        :rtype: bool
        """
        def my():
            l = len(pushed)
            if l == 0: return True
            s = {t:i for i,t in enumerate(pushed)}
            last = s[popped[0]]
            pushed[last] = -1
            i = 1
            
            while i < l:
                c = popped[i]
                ci = s[c]
                pre = last
                while pre >=0 and pushed[pre]==-1:
                    pre -= 1
                if ci==pre or ci>last:
                    last = ci
                    i += 1
                    pushed[ci] = -1
                    continue
                else:
                    return False
        def simulate():
            l = len(pushed)
            if l==0: return True
            stack = []
            i = j = 0
            while i<l and j<l:
                while i<l and pushed[i]!=popped[j]:
                    stack.append(pushed[i])
                    i += 1
                i += 1
                j += 1
                while stack and j>=0 and stack[-1]==popped[j]:
                    j += 1
                    stack.pop()
            if i==j and j==l:
                return True
            else:
                return False
        return simulate()

    @classmethod
    def verifyPostorder(self, postorder):
        """
        :type postorder: List[int]
        :rtype: bool
        """
        stack, root = [], float('+inf')
        for n in postorder[::-1]:
            if n > root:
                return False
            while stack and stack[-1] > n:# find root
                root = stack.pop()
            stack.append(n)
        return True


        import sys
        max_int = sys.maxsize
        bt_n = 1
        pl = len(postorder)
        while bt_n<pl: bt_n<<=1
        bt = [] # [max, min]
        for i in range(bt_n*2-1):
            bt.append([-max_int, max_int])
        def update(i, k):
            i += bt_n-1
            bt[i][0] = bt[i][1] = k
            while i>0:
                i = (i-1) >> 1
                l = (i<<1) + 1
                r = l + 1
                bt[i][0] = max(bt[l][0], bt[r][0])
                bt[i][1] = min(bt[l][1], bt[r][1])
        def query(a, b, i, l, r):
            if a<=l and r<=b:
                return bt[i]
            if r<=a or l>=b:
                return -max_int, max_int
            lre = query(a, b, 2*i+1, l, (l+r)>>1)
            rre = query(a, b, 2*i+2, (l+r)>>1, r)
            return max(lre[0], rre[0]), min(lre[1], rre[1])
        for i,t in enumerate(postorder):
            update(i, t)
        def dfs(l, r):
            if r<=l:
                return True
            for ln in range(r-l+1):
                if ln==0:
                    lmx = -max_int
                else:
                    lmx = query(l, l+ln, 0, 0, bt_n)[0]
                if ln==r-l:
                    rmn = max_int
                else:
                    rmn = query(l+ln, r, 0, 0, bt_n)[1]
                if lmx<postorder[r] and postorder[r]<rmn and dfs(l, l+ln-1) and dfs(l+ln, r-1):
                    return True
            return False
        res = dfs(0, len(postorder)-1)
        return res


            







a = [3,4,5,1,2]
b = [4, 1]
a = Codec.deserialize(a)
b = Codec.deserialize(b)
a = '(()'
res = Solution.verifyPostorder([1,3,2,6,5])
print(res)

opts = ["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
nums = [[],[1],[2],[],[3],[]]
solve = MedianFinder()
for i in range(1, len(opts)):
    if opts[i] == 'addNum':
        solve.addNum(nums[i][0])
    else:
        print(solve.findMedian())