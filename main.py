# import sys
# input_stream = open('input.txt', 'r')
# sys.stdin = input_stream

from functions import Solution
from datetime import datetime
from copy import deepcopy
solve = Solution()

a = solve.deserialize('[1,0]')
start = datetime.now()

res = solve._isSymmetric(
    a
)
print(res)

print(datetime.now() - start)