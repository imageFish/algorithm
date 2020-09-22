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
                    if data[i]!=None:
                        n.left = TreeNode(data[i])
                    i += 1
                    if data[i]!=None:
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


s = 'mississippi'
p = 'mis*is*p*.'
s = 'bbbba'
p = '.*a*a'
res = Solution.isMatch(s, p)
print(res)

opts = ["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
nums = [[],[1],[2],[],[3],[]]
solve = MedianFinder()
for i in range(1, len(opts)):
    if opts[i] == 'addNum':
        solve.addNum(nums[i][0])
    else:
        print(solve.findMedian())