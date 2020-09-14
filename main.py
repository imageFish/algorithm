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

res = solve.containsNearbyAlmostDuplicate(
    [10, 9, 8, 7, 6, 5, 4, -10, 1,5,9,1,5,9], 4, 0
)
print(res)

print(datetime.now() - start)