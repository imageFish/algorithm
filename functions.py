import collections
from copy import deepcopy


class Solution:
    # def findNumberIn2DArray(self, matrix: list[list[int]], target: int) -> bool:
    def findNumberIn2DArray(self, matrix, target):
        def binary_search(nums, target):
            r = len(nums) - 1
            l = 0
            while True:
                m = int((l + r) / 2)
                if nums[m] <= target:
                    l = m
                else:
                    r = m
                if r - l <= 1:
                    if nums[r] <= target:
                        return r
                    return l
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        max_len = len(matrix[0])
        for row in matrix:
            idx = binary_search(row[:max_len], target)
            if row[idx] == target:
                return True
            if idx == 0 and row[idx] > target:
                return False
            max_len = min(idx+1, max_len)
        return False

    # def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
    def buildTree(self, preorder, inorder):
        node2idx = {n: idx for idx, n in enumerate(inorder)}

        def build_child_tree(pre_l, pre_r, in_l, in_r):
            root = TreeNode(preorder[pre_l])
            root_idx = node2idx[preorder[pre_l]]

            lc_len = 0
            if root_idx != in_l:
                lc_len = root_idx - in_l
                root.left = build_child_tree(
                    pre_l+1, pre_l+lc_len, in_l, root_idx-1)
            if root_idx != in_r:
                rc_len = in_r - root_idx
                root.right = build_child_tree(
                    pre_l+lc_len+1, pre_l+lc_len+rc_len, root_idx+1, in_r)
            return root
        if len(preorder) == 0:
            return None
        return build_child_tree(0, len(preorder)-1, 0, len(inorder)-1)

    # def minArray(self, numbers: List[int]) -> int:
    # 熟练使用二分条件
    def minArray(self, numbers):
        def binary_search(x):
            l = 0
            r = len(x) - 1
            while True:
                m = int(l + (r-l)/2)
                if x[m] < x[r]:
                    r = m
                elif x[m] > x[r]:
                    if l == m:
                        l += 1
                    else:
                        l = m
                else:
                    r -= 1
                if r <= l:
                    break
            return x[r]
        return binary_search(numbers)
    # def exist(self, board: List[List[str]], word: str) -> bool:
    # deep first or

    def exist(self, board, word):
        def df(m, n, idx):
            if idx == len(word):
                return True
            if m < 0 or m >= M or n < 0 or n >= N or board[m][n] != word[idx]:
                return False
            tmp, board[m][n] = board[m][n], '/'
            res = df(m+1, n, idx+1) or df(m-1, n, idx +
                                          1) or df(m, n+1, idx+1) or df(m, n-1, idx+1)
            board[m][n] = tmp
            return res
        M, N = len(board), len(board[0])
        for i in range(len(board)):
            for j in range(len(board[0])):
                if df(i, j, 0):
                    return True
        return False

    # def movingCount(self, m: int, n: int, k: int) -> int:
    def movingCount(self, m, n, k):
        def num_sum(n):
            res = 0
            while n != 0:
                m = n % 10
                q = n // 10
                res += m
                n = q
            return res
        flags = []
        for _ in range(m):
            flags.append([0]*n)
        cnt = 0
        s = set()

        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or num_sum(i) + num_sum(j) > k or flags[i][j] == 1:
                return
            # flags[i][j] = 1
            nonlocal cnt
            cnt += 1
            flags[i][j] = 1
            dfs(i+1, j)
            dfs(i-1, j)
            dfs(i, j+1)
            dfs(i, j-1)
        dfs(0, 0)
        # cnt = sum([sum(t) for t in flags])
        return cnt

    # def isMatch(self, s: str, p: str) -> bool:
    def isMatch(self, s, p):
        s = '#' + s
        p = '#' + p
        m, n = len(s), len(p)
        dp = [[False]*n for _ in range(m)]
        dp[0][0] = True
        for i in range(m):
            for j in range(1, n):
                if i == 0:  # 初始化dp0
                    dp[i][j] = j > 1 and p[j] == '*' and dp[i][j-2]
                elif p[j] in ['.', s[i]]:
                    dp[i][j] = dp[i-1][j-1]
                elif p[j] == '*':  # 匹配0个或者1个， 1个时递归n个
                    dp[i][j] = j > 1 and dp[i][j-2] or p[j -
                                                         1] in [s[i], '.'] and dp[i-1][j]
        return dp[-1][-1]
        # def binary_search(s, p):
        #     if len(s) == 0 and len(p) == 0:
        #         return True
        #     if len(p) == 0 and len(s) != 0:
        #         return False

        #     if len(p) > 1 and p[1] == '*':
        #         if binary_search(s, p[2:]):
        #             return True
        #         i = 0
        #         while True:
        #             if i >= len(s):
        #                 break
        #             # if (p[0] != '.' and s[i] != p[0]) or (p[0] == '.' and s[i] != s[0]):
        #             if p[0] != '.' and s[i] != p[0]:
        #                 break
        #             if binary_search(s[i+1:], p[2:]):
        #                 return True
        #             i += 1
        #         return False
        #     else:
        #         if len(s) == 0:
        #             return False
        #         if p[0] == '.':
        #             return binary_search(s[1:], p[1:])
        #         else:
        #             if p[0] != s[0]:
        #                 return False
        #             else:
        #                 return binary_search(s[1:], p[1:])

        # return binary_search(s, p)

    def serialize1(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        par_layer = [root]
        flag = False if root == None else True
        res = []
        while flag:
            tmp = []
            flag = False
            for node in par_layer:
                if node == None:
                    res.append('null')
                    tmp.extend([None, None])
                else:
                    res.append(str(node.val))
                    tmp.extend([node.left, node.right])
                    if node.left != None or node.right != None:
                        flag = True
            par_layer = tmp
        res = '[' + ','.join(res) + ']'
        return res

    def deserialize1(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        data = data[1:-1].split(',')
        nodes = [None] * len(data)
        for i in range(len(data)):
            if data[i] == 'null':
                continue
            node = TreeNode(int(data[i]))
            nodes[i] = node
            if i == 0:
                continue
            if i % 2 == 0:
                idx = i // 2 - 1
                if nodes[idx] == None:
                    break
                nodes[idx].right = node
            else:
                idx = (i + 1) // 2 - 1
                if nodes[idx] == None:
                    break
                nodes[idx].left = node
        if len(nodes) == 0:
            nodes.append(None)
        return nodes[0]

    def serialize(self, root):
        if not root:
            return "[]"
        queue = collections.deque()
        queue.append(root)
        res = []
        while queue:
            node = queue.popleft()
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                res.append("null")
        return '[' + ','.join(res) + ']'

    def deserialize(self, data):
        if data == "[]":
            return
        vals, i = data[1:-1].split(','), 1
        root = TreeNode(int(vals[0]))
        queue = collections.deque()
        queue.append(root)
        while queue and i < len(vals):
            node = queue.popleft()
            if vals[i] != "null":
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1
            if vals[i] != "null":
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1
        return root

    # def isSymmetric(self, root: TreeNode) -> bool:
    def isSymmetric(self, root):
        if root == None:
            return True
        nodes = [root]
        while len(nodes) != 0:
            childs = []
            next_nodes = []
            for n in nodes:
                if n.left != None:
                    next_nodes.append(n.left)
                if n.right != None:
                    next_nodes.append(n.right)
                childs.extend([n.left, n.right])
            for i in range(len(childs)//2):
                if not childs[i] or not childs[-1-i]:
                    if childs[i] != childs[-1-i]:
                        return False
                    else:
                        continue
                if childs[i].val != childs[-1-i].val:
                    return False
            nodes = next_nodes
        return True

    def spiralOrder(self, matrix):
        m = len(matrix)
        n = len(matrix[0]) if m > 0 else 0
        res = []
        while m > 0 and n > 0:
            res.extend(matrix[0])
            matrix = matrix[1:]
            m -= 1
            if m == 0:
                break
            for t in range(m):
                res.append(matrix[t][-1])
                matrix[t] = matrix[t][:-1]
            n -= 1
            if n == 0:
                break
            res.extend(matrix[-1][::-1])
            matrix = matrix[:-1]
            m -= 1
            if m == 0:
                break
            for t in range(m-1, -1, -1):
                res.append(matrix[t][0])
                matrix[t] = matrix[t][1:]
            n -= 1
            if n == 0:
                break
        return res

    def validateStackSequences(self, pushed, popped):
        """
        :type pushed: List[int]
        :type popped: List[int]
        :rtype: bool
        """
        order_set = {t: i for i, t in enumerate(pushed)}
        order = [order_set[t] for t in popped]
        stack = [-1]
        m = -1
        for t in order:
            if t > stack[-1]:
                for i in range(m+1, t):
                    stack.append(i)
                m = max(m, t)
            elif t != stack[-1]:
                return False
            else:
                stack.pop()
        return True

    def verifyPostorder(self, postorder):
        stack, root = [], float('+inf')
        for n in postorder[::-1]:
            if n > root:
                return False
            while stack and stack[-1] > n:
                root = stack.pop()
            stack.append(n)
        return True

        def recurse(orders):
            if len(orders) <= 1:
                return True
            root = orders[-1]
            orders = orders[:-1]
            end = len(orders)
            l = orders[0]
            r = orders[-1]
            min_vals = []
            max_vals = []
            for i in range(end):
                if orders[i] >= l:
                    l = orders[i]
                max_vals.append(l)
                if orders[-1-i] <= r:
                    r = orders[-1-i]
                min_vals.append(r)
            min_vals = min_vals[::-1]
            for i in range(end+1):
                if i == 0:
                    l = root - 1
                else:
                    l = max_vals[i-1]
                if i == end:
                    r = root + 1
                else:
                    r = min_vals[i]
                if l < root and root < r and recurse(orders[:i]) and recurse(orders[i:]):
                    return True
            return False
        return recurse(postorder)

    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        res = []
        if root == None:
            return res
        # if root.val > sum:
        #     return res
        res = [[root.val]]
        nodes = [root]
        cur_sum = [root.val]
        res_return = []
        while nodes:
            _nodes = []
            _cur_sum = []
            _res = []
            for i, n in enumerate(nodes):
                if n.left != None:
                    _nodes.append(n.left)
                    _cur_sum.append(n.left.val + cur_sum[i])
                    _res.append(res[i] + [n.left.val])
                if n.right != None:
                    _nodes.append(n.right)
                    _cur_sum.append(n.right.val + cur_sum[i])
                    _res.append(res[i] + [n.right.val])
                if n.left == None and n.right == None and cur_sum[i] == sum:
                    res_return.append(res[i])
            nodes = _nodes
            cur_sum = _cur_sum
            res = _res
        return res_return
    
    def permutation(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def recurse(s):
            if len(s) == 0:
                return []
            if len(s) == 1:
                return [s]
            res = []
            char = set()
            for i in range(len(s)):
                if s[i] in char:
                    continue
                char.add(s[i])
                _res = self.permutation(s[:i] + s[i+1:])
                for j in range(len(_res)):
                    _res[j] = s[i] + _res[j]
                res.extend(_res)
            return res
        res = recurse(s)
        return res

    def getLeastNumbers(self, arr, k):
        import heapq
        res = heapq.nsmallest(k, arr)
        return res

        import heapq
        # import _heapq as heapq
        l = len(arr)
        k = min(l, k)
        if k == 0:
            return []
        res = deepcopy(arr[:k])
        heapq._heapify_max(res)
        for i in range(k, l):
            if arr[i] < res[0]:
                res[0] = arr[i]
                heapq._siftup_max(res, 0)
        return res

    def countDigitOne(self, n):
        """
        :type n: int
        :rtype: int
        """
        high = n // 10
        low = 0
        digit = 1
        cur = n % 10
        res = 0
        while high != 0 or cur != 0:
            if cur == 0:
                res += high * digit
            elif cur == 1:
                res += high * digit + low + 1
            else:
                res += (high + 1) * digit
            digit *= 10
            cur = high % 10
            high = high // 10
            low = n % digit
        return res

    def minNumber(self, nums):
        def binary_sort(l, r):
            if r-l < 1:
                return
            m = (l+r) >> 1
            m += 1
            binary_sort(l, m-1) 
            binary_sort(m, r)
            tmp = []
            tmp_l = r-l+1
            _l, _m = l, m 
            while _l < m and _m < r+1:
                if nums[_l] + nums[_m] < nums[_m] + nums[_l]:
                    tmp.append(nums[_l])
                    _l += 1
                else:
                    tmp.append(nums[_m])
                    _m += 1
            if _l < m:
                tmp.extend(nums[_l:m])
            if _m < r+1:
                tmp.extend(nums[_m:r+1])
            nums[l:r+1] = tmp
        def quick_sort(l, r):
            if r-l < 1:
                return
            _l, _r = l, r 
            while _l < _r:
                while _l < _r and nums[l] + nums[_r] <= nums[_r] + nums[l]:
                    _r -= 1
                while _l < _r and nums[l] + nums[_l] >= nums[_l] + nums[l]:
                    _l += 1
                nums[_l], nums[_r] = nums[_r], nums[_l]
            nums[_l], nums[l] = nums[l], nums[_l]
            quick_sort(l, _l-1)
            quick_sort(_l+1, r)

        nums = [str(i) for i in nums]
        # binary_sort(0, len(nums)-1)
        quick_sort(0, len(nums)-1)
        return ''.join(nums)

    def translateNum(self, num):
        """
        :type num: int
        :rtype: int
        """
        num = str(num)
        dp = [0] * (len(num)+1)
        dp[1] = 1
        dp[0] = 1
        for i in range(1, len(num)):
            dp[i+1] = dp[i]
            if num[i-1] == '0':
                continue
            t = int(num[i-1:i+1])
            if 0<=t and t<=25:
                dp[i+1] += dp[i-1]
        return dp[-1]

    def maxValue(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        dp_last = [0] * (n+1)
        for i in range(m):
            dp = [0] * (n+1)
            for j in range(n):
                dp[j+1] = max(dp[j], dp_last[j+1]) + grid[i][j]
            dp_last = dp
        return dp_last[-1]

    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        # char = {}
        max_len = 0
        res = 0
        for i in range(len(s)):
            j = 0
            while j < max_len:
                if s[i-1-j] == s[i]:
                    break
                j += 1
            if j == max_len-1 and s[i-1-j] != s[i]:
                max_len += 1
            else:
                max_len = j+1
            res = max(max_len, res)
        return res
    
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [1]
        a = b = c = 0
        for i in range(1, n):
            _a = dp[a] * 2
            _b = dp[b] * 3
            _c = dp[c] * 5
            dp.append(min(_a, _b, _c))
            if dp[-1] == _a: a += 1
            if dp[-1] == _b: b += 1
            if dp[-1] == _c: c += 1
        return dp[-1]

    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def bst(l, r, t):
            if r-l < 0:
                return 0 
            _l, _r = l, r
            while r-l > 1:
                m = (l+r) >> 1
                if t <= nums[m]: r = m 
                else: l = m
            if nums[l] >= t:
                return 0
            if nums[r] < t:
                return r-_l+1
            if nums[r] == t:
                return r-_l
            return l-_l+1
        def pre_sum():
            tmp = []
            for n in nums[::-1]:
                l, r = 0, len(tmp)
                while r-l < 1:
                    m = (l+r) >> 1
                    if tmp[m] >= nums[i]: r=m
                    else: l=m
                if 
        def quick_sort(l, r):
            if r-l < 1:
                return 0
            m = (l+r) >> 1
            res = quick_sort(l, m) 
            res += quick_sort(m+1, r)
            tmp = []
            _l, _r = l, m+1
            while _l <= m and _r <= r:
                if nums[_l] <= nums[_r]:
                    tmp.append(nums[_l])
                    _l += 1
                else:
                    res += m+1-_l
                    tmp.append(nums[_r])
                    _r += 1
            if _l <= m:
                tmp.extend(nums[_l:m+1])
            if _r <= r:
                tmp.extend(nums[_r:r+1])
            nums[l:r+1] = tmp
            return res
        return quick_sort(0, len(nums)-1)
            






class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class CQueue:

    def __init__(self):
        self.query = []
        self.head = self.tail = -1

    def appendTail(self, value: int) -> None:
        if self.tail == len(self.query) - 1:
            self.query.append(value)
        else:
            self.query[self.tail] = value
        if self.head == -1:
            self.head += 1
        self.tail += 1

    def deleteHead(self) -> int:
        if self.tail == self.head and self.tail == -1:
            return -1
        res = self.query[self.head]
        if self.tail == self.head:
            self.tail = self.head = -1
        else:
            self.head += 1
        return res
