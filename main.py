# import sys
# input_stream = open('input.txt', 'r')
# sys.stdin = input_stream

from functions import Solution
from datetime import datetime
solve = Solution()
a = '[-2,null,-3]'
# a = solve.getLeastNumbers(a)

start = datetime.now()

res = solve.getLeastNumbers([0,1,1,2,4,4,1,3,3,2], 6)
print(res)
print(datetime.now() - start)