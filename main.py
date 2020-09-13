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

res = solve.findCriticalAndPseudoCriticalEdges(
    n = 5, edges = [[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]]
)
print(res)

print(datetime.now() - start)