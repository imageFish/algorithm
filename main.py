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

res = solve.accountsMerge(
    [["David","David0@m.co","David4@m.co","David3@m.co"],["David","David5@m.co","David5@m.co","David0@m.co"],["David","David1@m.co","David4@m.co","David0@m.co"],["David","David0@m.co","David1@m.co","David3@m.co"],["David","David4@m.co","David1@m.co","David3@m.co"]]
)
print(res)

print(datetime.now() - start)