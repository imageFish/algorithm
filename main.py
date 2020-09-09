# import sys
# input_stream = open('input.txt', 'r')
# sys.stdin = input_stream

from functions import Solution
from datetime import datetime
from copy import deepcopy
solve = Solution()

start = datetime.now()

res = solve.add(
    3, -1
)
print(res)

print(datetime.now() - start)