class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def cuttingRope(self, n: int) -> int:
        if n == 2:
            return 1
        if n == 3:
            return 2
        if n == 4:
            return 4
        s, y = divmod(n, 3)
        if y == 1:
            s = s - 1
            y = 4
        res = 1
        for i in range(s):
            res = (res * 3) % 1000000007
        if y == 2:
            res = (res * 2) % 1000000007
        if y == 4:
            res = (res * 4) % 1000000007
        return res
    def myPow(self, x: float, n: int) -> float:
        abs_n = abs(n)
        res = self.pow(x, abs_n)
        if res == 0:
            return 0.0
        if n < 0:
            res = 1 / res
        return res
    def pow(self, x, n):
        ps = {0:x}
        bn = ''
        _n = n
        k = 1
        while _n != 0:
            _n, _m = divmod(_n, 2)
            bn = str(_m) + bn
            ps[k] = ps[k-1] * ps[k-1]
            k += 1
        res = 1
        bn = bn[::-1]
        for c_idx, c in enumerate(bn):
            if c == '1':
                res = res * ps[c_idx]
        return res

    def findNumberIn2DArray(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        def binary_search(nums, t):
            assert len(nums) > 0
            l, r = 0, len(nums) - 1
            while l <= r:
                mid = int((l+r)/2)
                if mid == l:
                    if nums[r] <= t:
                        return r
                    elif nums[l] <=t:
                        return l
                    else:
                        return -1
                if nums[mid] < t:
                    l = mid
                else:
                    r = mid
        
        M = len(matrix)
        if M == 0:
            return False
        N = len(matrix[0])
        if N == 0:
            return False
        m, n = 0, N
        while True:
            n0 = binary_search(matrix[m][:n+1], target)
            if n0 == -1:
                return False
            if matrix[m][n0] == target:
                return True
            n = min(n, n0)
            m += 1
            if m == M:
                return False
    

    inorder_set = None
    inorder = None
    def buildTree(self, preorder, inorder, l=None, r=None):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if len(preorder) == 0:
            return None
        root = TreeNode(preorder[0])

        if l == None and r==None:
            self.inorder = inorder
            self.inorder_set = {node:idx for idx, node in enumerate(inorder)}
            l = 0
            r = len(inorder)
        idx = self.inorder_set[root.val]
        assert l <= idx and idx < r
        left_inorder = self.inorder[l:idx]
        right_inorder = self.inorder[idx+1:r]

        if len(left_inorder) != 0:
            left_preorder = preorder[1:1+len(left_inorder)]
            root.left = self.buildTree(left_preorder, [], l=l, r=idx)
        if len(right_inorder) != 0:
            right_preorder = preorder[1+len(left_inorder):]
            root.right = self.buildTree(right_preorder, [], l=idx+1, r=r)
        return root



    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        word_set = {}
        for i in range(len(board)):
            for j in range(len(board[0])):
                w = board[i][j]
                if w in word_set:
                    word_set[w].append([i, j])
                else:
                    word_set[w] = [[i, j]]
        M, N = len(board), len(board[0])
        if len(word) == 0:
            return True
        if len(word_set) == 0:
            return False
        def find_path(subword, used, pre_pos):
            if len(subword) == 0:
                return True
            for h, v in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                new_pos = [pre_pos[0] + h, pre_pos[1] + v]
                if new_pos[0] < 0 or new_pos[0] >= M or new_pos[1] < 0 or new_pos[1] >= N:
                    continue
                new_pos_str = '_'.join([str(t) for t in new_pos])
                if board[new_pos[0]][new_pos[1]] != subword[0]:
                    continue
                if new_pos_str in used:
                    continue
                used[new_pos_str] = 1
                if find_path(subword[1:], used, new_pos):
                    return True
                used.pop(new_pos_str)
            return False
        if word[0] not in word_set:
            return False
        for i,j in word_set[word[0]]:
            used = {
                str(i)+'_'+str(j): 1
            }
            if find_path(word[1:], used, [i,j]):
                return True
        return False




