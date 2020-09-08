# import sys
# input_stream = open('input.txt', 'r')
# sys.stdin = input_stream

from functions import Solution
from datetime import datetime
from copy import deepcopy
solve = Solution()

start = datetime.now()

res = solve.singleNumber(
    [9,1,7,9,7,9,7]
)
print(res)

print(datetime.now() - start)