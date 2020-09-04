# import sys
# input_stream = open('input.txt', 'r')
# sys.stdin = input_stream

from functions import Solution
from datetime import datetime
solve = Solution()
a = '[-2,null,-3]'
# a = solve.getLeastNumbers(a)

start = datetime.now()

res = solve.minNumber(
    [3,30,34,5,9]
)
print(res)
print(datetime.now() - start)