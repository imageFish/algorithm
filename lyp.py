import sys
n = int(sys.stdin.readline().strip())
for i in range(n):
    equ = sys.stdin.readline().strip()
    left = [i for i,s in enumerate(equ) if s=='(']
    left = left[::-1]
    right = [i for i,s in enumerate(equ) if s==')']
    if len(left) != len(right):
        print('invalid')
        continue
    res = None
    invalid = False
    il, ir = -1, -1
    for l, r in zip(left, right):
        if il == -1:
            s = s[l+1:r]
            sl = s.split()
            opt = None,
            n1, n2 = None, None
            if len(sl) == 3:
                opt = sl[0]
                try:
                    n1 = int(sl[1])
                    n2 = int(sl[2])
                except Exception as identifier:
                    invalid = True
            elif len(sl) == 2:
                opt = sl[0][1]
                try:
                    n1 = int(sl[0][1:])
                    n2 = int(sl[2])
                except Exception as identifier:
                    invalid = True
            else:
                invalid = True
            if invalid:
                break
            if opt == '+':
                res = n1 + n2
            elif opt == '-':
                res = n1 - n2
            elif opt == '*':
                res = n1 * n2
            else:
                invalid = True
        else:
            s = s[l+1:il] + s[ir+1:r]
            sl = s.split()
            if len(sl) == 2:
                opt = nu