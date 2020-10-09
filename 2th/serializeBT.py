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

    @classmethod
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        lis = [0] * len(nums)
        res = len(nums) - 1
        def bs(k):
            l, r = res, len(lis)-1
            while l < r:
                m = (l+r+1) >> 1
                if lis[m] <= k:
                    r = m - 1
                else:
                    l = m
            return r
        for k in nums:
            idx = bs(k)
            lis[idx] = k
            if idx == res:
                res -= 1
        rres = len(nums)-1-res
        print(rres)
    
    @classmethod
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        from copy import deepcopy
        used = []
        res = []
        for _ in range(n):
            used.append([0]*n)
        
        def label(i, j, lab):
            for k in range(1, n):
                flag = False
                _i = i + k
                if _i < n: 
                    used[_i][j] += lab 
                    flag = True
                    _j = j + k
                    if _j < n: used[_i][_j] += lab
                    _j = j - k
                    if _j >= 0: used[_i][_j] += lab
                _i = i - k
                if _i >= 0: 
                    used[_i][j] += lab
                    flag = True
                    _j = j + k
                    if _j < n: used[_i][_j] += lab
                    _j = j - k
                    if _j >= 0: used[_i][_j] += lab
                if not flag:
                    break
        def dfs(i, _res):
            if i == n:
                tmp = [t for t in _res.split(',') if len(t)!=0]
                res.append(tmp)
                return
            for j in range(n):
                if used[i][j] > 0:
                    continue
                label(i, j, 1)
                tmp = ['.']*n
                tmp[j] = 'Q'
                dfs(i+1, _res+','+''.join(tmp))
                label(i, j, -1)
        dfs(0, '')
        return res
    
    @classmethod
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        from copy import deepcopy
        used = []
        res = 0
        for _ in range(n):
            used.append([0]*n)
        
        def label(i, j, lab):
            for k in range(1, n):
                flag = False
                _i = i + k
                if _i < n: 
                    used[_i][j] += lab 
                    flag = True
                    _j = j + k
                    if _j < n: used[_i][_j] += lab
                    _j = j - k
                    if _j >= 0: used[_i][_j] += lab
                _i = i - k
                if _i >= 0: 
                    used[_i][j] += lab
                    flag = True
                    _j = j + k
                    if _j < n: used[_i][_j] += lab
                    _j = j - k
                    if _j >= 0: used[_i][_j] += lab
                if not flag:
                    break
        def dfs(i):
            if i == n:
                nonlocal res
                res += 1
                return
            for j in range(n):
                if used[i][j] > 0:
                    continue
                label(i, j, 1)
                dfs(i+1)
                label(i, j, -1)
        dfs(0)
        return res


    @classmethod
    def findMinStep(self, board, hand):
        """
        :type board: str
        :type hand: str
        :rtype: int
        """
        def eliminate(s):
            mn = len(s)
            for i in range(len(s)):
                j = i
                while j<len(s) and s[j]==s[i]:
                    j += 1
                if j-i >= 3:
                    tmp = s[:i] + s[j:]
                    _mn = eliminate(tmp)
                    mn = min(mn, _mn)
            return mn
        def dfs(board, hand):
            mx = -1e10
            i = 0
            while i < len(board):
                j = i
                while j<len(board) and board[j]==board[i]:
                    j += 1
                z = j-i
                if z >= 3:
                    i = j
                    continue
                t = 3 - z
                c = board[i]
                if hand[c] >= t:
                    hand[c] -= t
                    # tmp = eliminate(board[:i] + c*t + board[i:])
                    tmp = board[:i] + board[j:]
                    left_len = eliminate(tmp)
                    if left_len==0:
                        _mx = sum(hand.values())
                    else:
                        _mx = dfs(tmp, hand)
                    mx = max(_mx, mx)
                    hand[c] += t
                i = j
            return mx
        _hand = {'Y':0, 'W':0, 'R':0, 'B':0, 'G':0}
        for c in hand:
            _hand[c] += 1
        res = dfs(board, _hand)
        if res == -1e10:
            res = -1
        else:
            res = len(hand) - res
        return res
    
    @classmethod
    def findTheLongestSubstring(self, s):
        sl = len(s)
        pre = {}
        pre[0] = -1
        cur = 0
        res = -1
        for i, c in enumerate(s):
            if c == 'a':
                cur ^= 1
            elif c == 'e':
                cur ^= 2
            elif c == 'i':
                cur ^= 4
            elif c == 'o':
                cur ^= 8
            elif c == 'u':
                cur ^= 16
            if cur not in pre:
                pre[cur] = i
            else:
                res = max(res, i-pre[cur])
        return res

    @classmethod
    def minimumOperations(self, leaves):
        """
        :type leaves: str
        :rtype: int
        """
        ll = len(leaves)
        d0 = 0 if leaves[0]=='r' else 1
        d1 = d2 = ll
        for i in range(1, ll):
            _d0 = d0 if leaves[i]=='r' else d0+1
            _d1 = min(d0, d1)
            if leaves[i]!='y':
                _d1 += 1
            _d2 = min(d1, d2)
            if leaves[i]!='r':
                _d2 += 1
            d0, d1, d2 = _d0, _d1, _d2
        return d2
    
    @classmethod
    def multiply(self, num1, num2):
        # num1, num2 = num1[::-1], num2[::-1]
        num1 = [int(t) for t in num1]
        num2 = [int(t) for t in num2]
        l1, l2 = len(num1), len(num2)
        res = [0] * (l1+l2)
        for i in range(l1-1, -1, -1):
            for j in range(l2-1, -1, -1):
                a, b = num1[i], num2[j]
                t = a * b + res[i+j+1]
                t0 = t % 10
                t1 = t // 10
                res[i+j+1] = t0
                res[i+j] += t1
        i = 0
        while i < l1+l2-1 and res[i]==0:
            i += 1
        res = ''.join([str(t) for t in res[i:]])
        return res
    
    @classmethod
    def restoreIpAddresses(self, s):
        res = []
        l = len(s)
        t = 0
        for i in range(1, 4):
            t0 = t + i
            if (t0>l) or (s[t]=='0' and i>1):
                break
            a0 = int(s[t:t0])
            if not (0<=a0 and a0<=255):
                break
            for j in range(1, 4):
                t1 = t0 + j
                if (t1>l) or (s[t0]=='0' and j>1):
                    break
                a1 = int(s[t0:t1])
                if not (0<=a1 and a1<=255):
                    break
                for k in range(1, 4):
                    t2 = t1 + k
                    if (t2>=l) or (s[t1]=='0' and k>1):
                        break
                    a2 = int(s[t1:t2])
                    if not (0<=a2 and a2<=255):
                        break
                    if s[t2]=='0' and t2!=l-1:
                        continue
                    a3 = int(s[t2:])
                    if not(0<=a3 and a3<=255):
                        continue
                    res.append([a0, a1, a2, a3])
        res = ['.'.join([str(t) for t in i]) for i in res]
        return res

    @classmethod
    def minJump(self, jump):
        def dp():
            res = n = len(jump)
            f = [n]*n
            f[0] = 0
            mx_dis = [0]*n
            mx_step = 0
            for i in range(n):
                if i > mx_dis[mx_step]:
                    mx_step += 1
                f[i] = min(f[i], mx_step+1)
                tmp = i + jump[i]
                if tmp >= n:
                    res = min(res, f[i]+1)
                else:
                    f[tmp] = min(f[tmp], f[i]+1)
                    mx_dis[f[i]+1] = max(mx_dis[f[i]+1], tmp)
            return res
        def bfs():
            n = len(jump)
            visited = [False]*n
            visited[0] = True
            stack = [0] # visited idxs
            res = 0
            left = 1
            while stack:
                _stack = []
                res += 1
                for idx in stack:
                    tmp = idx + jump[idx]
                    if tmp >= n:
                        return res
                    if not visited[tmp]:
                        visited[tmp] = True
                        _stack.append(tmp)
                    for j in range(left, idx):
                        if not visited[j]:
                            visited[j] = True
                            _stack.append(j)
                    left = max(left, idx+1) # it's possible that left is greater than idx
                stack = _stack
            return -1

        return bfs()

    @classmethod
    def fourSum(self, nums, target):
        sums = []
        nl = len(nums)
        for i in range(nl):
            for j in range(i+1, nl):
                sums.append([nums[i]+nums[j], i, j])
        sums.sort()
        def bs(t, l=0, r=len(sums)):
            while l < r:
                m = (l+r) >> 1
                if sums[m][0] >= t:
                    r = m
                else:
                    l = m + 1
            return l
        res = []
        res_set = set()
        for s0_i, s0 in enumerate(sums):
            s1 = target - s0[0]
            idx = bs(s1, l=s0_i+1)
            j = idx
            while j<len(sums) and sums[j][0]==s1:
                tmp = [s0[1], s0[2], sums[j][1], sums[j][2]]
                if len(set(tmp)) == len(tmp):
                    tmp = [nums[t] for t in tmp]
                    tmp.sort()
                    k = '_'.join([str(t) for t in tmp])
                    if k not in res_set:
                        res.append(tmp)
                        res_set.add(k)
                j += 1
        return res

    @classmethod
    def threeSum(self, nums):
        nl = len(nums)
        nums.sort()
        res = []
        i = 0
        while i < nl:
            while i>0 and i<nl and nums[i]==nums[i-1]:
                i += 1
            l, r = i+1, nl-1
            while l < r:
                while l>i+1 and l<r and nums[l]==nums[l-1]:
                    l += 1
                while r > l and nums[l]+nums[r]>-nums[i]:
                    r -= 1
                if r==l: break
                if nums[l]+nums[r] == -nums[i]:
                    res.append([nums[i], nums
                    [l], nums[r]])
                l += 1
            i += 1
        return res

    @classmethod
    def search(self, nums, target):
        nl = len(nums)
        if nums[-1] < nums[0]:
            l, r = 0, nl-1
            while r-l>1:
                m = (l+r) >> 1
                if nums[m] > nums[l]:
                    l = m
                else:
                    r = m
            nums = nums[r:] + nums[:r]
            idx = r
        else:
            idx = 0
        l, r = 0, nl
        while l < r:
            m = (l+r) >> 1
            if nums[m] >= target:
                r = m
            else:
                l = m + 1
        if l!=nl and nums[l]==target:
            if l >= nl-idx:
                l -= nl-idx
            else:
                l += idx
            return l
        return -1
    
    @classmethod
    def findKthLargest(self, nums, k):
        import heapq
        res = heapq.nlargest(k, nums)
        return res[-1]

        arr = [nums[i] for i in range(k)]
        heapq.heapify(arr)
        for t in nums[k:]:
            if t > arr[0]:
                heapq.heappop(arr)
                heapq.heappush(arr, t)
        return arr[0]

    @classmethod
    def getPermutation(self, n, k):
        arr = [i+1 for i in range(n)]
        jc = [1]
        for i in range(1, n-1):
            jc.append(jc[-1]*(i+1))
        jc = jc[::-1]
        for i in range(n-1):
            p, m = k//jc[i], k%jc[i]
            if m==0:
                p -= 1
            arr[i], arr[p+i] = arr[p+i], arr[i]
            arr[i+1:] = sorted(arr[i+1:])
            if m==0:
                k = jc[i]
            else:
                k = m
        return ''.join([str(t) for t in arr])
    
    @classmethod
    def sortColors(self, nums):
        nl = len(nums)
        l, r = 0, nl-1
        while True:
            while l<r and nums[l]==0:
                l += 1
            while l<r and nums[r]!=0:
                r -= 1
            if l >= r:
                break
            nums[l], nums[r] = nums[r], nums[l]
        r = nl - 1
        while True:
            while l<r and nums[l]==1:
                l += 1
            while l<r and nums[r]!=1:
                r -= 1
            if l >= r:
                break
            nums[l], nums[r] = nums[r], nums[l]
        print(nums)
    
    @classmethod
    def findCircleNum(self, M):
        ml = len(M)
        uf = [i for i in range(ml)]
        for i in range(ml):
            for j in range(i+1, ml):
                if M[i][j]==1:
                    jp = j
                    while uf[jp] != jp:
                        jp = uf[jp]
                    ip = i
                    while uf[ip] != ip:
                        ip = uf[ip]
                    if jp != ip:
                        uf[jp] = ip
        visited = [False]*ml
        res = 0
        for i in range(ml-1, -1, -1):
            flag = False
            while uf[i] != i:
                if visited[i]:
                    flag = True
                    break
                visited[i] = True
                i = uf[i]
            if flag or visited[i]:
                continue
            visited[i] = True
            res += 1
        return res

    @classmethod
    def merge(self, intervals):
        intervals.sort()
        res = []
        il = len(intervals)
        i = 0
        while i < il:
            tmp = [intervals[i][0], intervals[i][1]]
            while i<il and tmp[1]>=intervals[i][0]:
                tmp[0] = min(tmp[0], intervals[i][0])
                tmp[1] = max(tmp[1], intervals[i][1])
                i += 1
            res.append(tmp)
        return res

    @classmethod
    def trap(self, height):
        hl = len(height)
        mx = mx_idx = -1
        for i in range(hl):
            if height[i] >= mx:
                mx_idx = i
                mx = height[i]
        i = 0
        res = 0
        while i < mx_idx:
            j = i
            while j<=mx_idx and height[j] <= height[i]:
                res += height[i] - height[j]
                j += 1
            i = j
        i = hl - 1
        while i > mx_idx:
            j = i
            while j >= mx_idx and height[j] <= height[i]:
                res += height[i] - height[j]
                j -= 1
            i = j
        return res

a = [3,4,5,1,2]
b = [4, 1]
a = Codec.deserialize(a)
b = Codec.deserialize(b)
a = '(()'
res = Solution.trap([0,1,0,2,1,0,1,3,2,1,2,1])
print(res)

opts = ["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
nums = [[],[1],[2],[],[3],[]]
solve = MedianFinder()
for i in range(1, len(opts)):
    if opts[i] == 'addNum':
        solve.addNum(nums[i][0])
    else:
        print(solve.findMedian())