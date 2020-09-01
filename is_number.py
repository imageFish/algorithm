class Solution:
    def isNumber(self, s: str) -> bool:
        # s = s.strip()
        return self.is_exp(s) or self.is_number_with_sign(s)[0]
    
    def is_exp(self, s):
        idxs = []
        for idx, i in enumerate(s):
            if i == 'e' or i == 'E':
                idxs.append(idx)
        if len(idxs) != 1:
            return False
        idx = idxs[0]
        v = s[:idx]
        e = s[idx+1:]
        v_flag = self.is_number_with_sign(v)
        if v_flag[0]:
            v_flag = True
        else:
            v_flag = False
        e_flag = self.is_number_with_sign(e)
        if e_flag[0] and e_flag[2] == -1:
            e_flag = True
        else:
            e_flag = False
        return e_flag and v_flag
    
    def is_number_with_sign(self, s):
        if len(s) == 0:
            return False, False, -1
        sign_flag = False
        if s[0] == '+' or s[0] == '-':
            sign_flag = True
            s = s[1:]
        if len(s) == 0:
            return False, False, -1
        float_flag = self.is_float(s)
        return float_flag[0] or self.is_int(s), sign_flag, float_flag[1]

    def is_float(self, s):
        idxs = []
        for idx, i in enumerate(s):
            if i == '.':
                idxs.append(idx)
        if len(idxs) != 1:
            return False, -1
        idx = idxs[0]
        if idx == 0:
            return self.is_int(s[1:]), idx
        if idx == len(s) - 1:
            return self.is_int(s[:idx]), idx
        return self.is_int(s[:idx]) and self.is_int(s[idx+1:]), idx

    def is_int(self, s):
        if len(s) == 0:
            return False
        for c in s:
            if c < '0' or c > '9':
                return False
        return True