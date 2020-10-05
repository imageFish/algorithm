def my():
    # a[i]==0时 不需要所以去掉
    import sys
    # sys.stdin = open('input.txt', 'r')
    n = int(sys.stdin.readline().strip())
    arr = [int(t) for t in sys.stdin.readline().strip().split()]
    dp = [[0, 0]] * (n+3) # 取arr[i]结尾的最大
    res = [0, 0]
    for i, t in enumerate(arr):
        l, r = t+dp[i][0], t+dp[i+1][0]
        if l == r:
            dp[i+3] = [l, min(dp[i][1], dp[i+1][1])+1]
        elif l > r:
            dp[i+3] = [l, dp[i][1]+1]
        else:
            dp[i+3] = [r, dp[i+1][1]+1]
        if t == 0: # 0 不需要取
            dp[i+3][1] -= 1
        if dp[i+3][0] > res[0]:
            res[0], res[1] = dp[i+3][0], dp[i+3][1]
        elif dp[i+3][0] == res[0]:
            res[1] = min(res[1], dp[i+3][1])
    print('{} {}'.format(*res))

def max_value_min_number():
    import sys
    sys.stdin = open('input.txt', 'r')
    n = eval(input())
    nums = [int(i) for i in sys.stdin.readline().split()]
    dp = [0 for _ in range(n+2)] # dp[i+2] >= dp[i+1] 
    # i之前的最大
    dpNum = [0 for _ in range(n+2)]
    num = 0
    for i in range(n):
        if dp[i+1] < dp[i]+nums[i]:
            dp[i+2] = dp[i]+nums[i]
            dpNum[i+2] = dpNum[i]+1
        else:
            dp[i+2] = dp[i+1]
            dpNum[i+2] = dpNum[i+1]
    print(dp[-1], dpNum[-1])
def note():
    import sys
    sys.stdin = open('input.txt', 'r')
    stack = []
    res = ''
    arr = sys.stdin.readline().strip()
    i = 0
    while i < len(arr):
        if stack:
            if arr[i] == ')':
                stack.pop()
            elif arr[i] == '(':
                stack.append('(')
            i += 1
            continue
        if arr[i] == '(':
            stack.append('(')
        elif arr[i] == '<':
            res = res[:-1]
        else:
            res += arr[i]
        i += 1
    print(res)


def lcs(y):
    dp = [0]*len(y)
    res = 0
    for t in y:
        l = 0
        r = res
        while l < r:
            m = (l+r) >> 1
            if dp[m] > t:
                r = m
            else:
                l = m+1
        dp[l] = t
        if l == res:
            res += 1
        # if not dp or i > dp[-1]:
        #     dp.append(i)
        # else:
        #     l, r = 0, len(dp)-1
        #     loc = r
        #     while l<=r:
        #         m = (l+r)>>1
        #         if dp[m] >= i:
        #             r = m-1
        #             loc = m
        #         else:
        #             l = m+1
        #     dp[loc] = i
    return res
import sys
sys.stdin = open('input.txt', 'r')
n = int(sys.stdin.readline().strip())
arr = [[int(t) for t in sys.stdin.readline().strip().split()] for _ in range(n)]
arr.sort()
y = [t[1] for t in arr]
print(lcs(y))