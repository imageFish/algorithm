class Solution:
    def strToInt(self, str: str) -> int:
        str = str.strip()
        s = ''
        for c in str:
            if c == '+' or c == '-':
                if len(s) == 0:
                    s += c
                    continue
                else:
                    break
            if c < '0' or c > '9':
                break
            s += c
        if len(s) == 0:
            return 0
        res = 0
        int_min = - 2**31
        int_max = 2**31 - 1
        positive = True
        max_num = int_max
        if s[0] == '+' or s[0] == '-':
            if s[0] == '-':
                positive = False
                max_num = -int_min
            s = s[1:]
        overflow = False
        for c in s:
            c = int(c)
            tmp = max_num // 10
            if res > tmp:
                overflow = True
                break
            res = res * 10
            if int_max - res < c:
                overflow = True
                break
            res = res + c
        if positive:
            if overflow:
                return int_max
            else:
                return res
        else:
            if overflow:
                return int_min
            else:
                return - res
