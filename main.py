# import sys
# input_stream = open('input.txt', 'r')
# sys.stdin = input_stream

from functions import Solution, Codec
from datetime import datetime
from copy import deepcopy
solve = Solution()

a = solve.deserialize('[]')
start = datetime.now()

# codec = Codec()
# res = codec.deserialize(codec.serialize(a))

res = solve.nextPermutation(
    [3,2,1]
)
print(res)

print(datetime.now() - start)