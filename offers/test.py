链接：https://www.nowcoder.com/questionTerminal/2ea45460386f413d9ed6cf532ffdcf28?f=discussion
来源：牛客网

class Solution:
    def findMinStep(self, board: str, cost:int) -> int:
        def elimination(s, start):
            if start < 0:
                return s
            # 从start开始对s进行消去
            len_s = len(s)
            if len_s <= 2&nbs***bsp;start==len_s-1:
                return s
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
     
board = input()
cost = int(input())
k = Solution().findMinStep(board,cost)
for i in k:
    print(i)