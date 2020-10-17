class Solution:
    @classmethod
    def abs_sort(self , input_a ):
        # write code here
        a = [abs(t) for t in input_a]
        al = len(a)
        idx = 0
        am = a[0]
        for i in range(1, al):
            if a[i]<am:
                am = a[i]
                idx = i
        l, r = idx-1, idx+1
        res = [input_a[idx]]
        while l>-1 and r<al:
            if a[l] <= a[r]:
                res.append(input_a[l])
                l -= 1
            else:
                res.append(input_a[r])
                r += 1
        if l>-1:
            res.extend(input_a[:l+1][::-1])
        if r<al:
            res.extend(input_a[r:])
        return res



        il = len(input_a)
        l, r = 0, il-1
        while k < r:
            m = (l+r) >> 1
            if m == l:
                break
            am = abs(input_a[m])
            ml, mr = m
            while ml>-1 and abs(input_a[ml])==am:
                ml -= 1
            while mr<il and abs(input_a[mr])==am:
                mr += 1
            aml, amr = abs(aml), abs(am), abs(amr)
            if aml<=am and am<=amr:
                m 
        return 0
    
    @classmethod
    def dotMultiply(self , A , B ):
        # write code here
        res = []
        am, an, bm, bn = len(A), len(A[0]), len(B), len(B[0])
        res = []
        for _ in range(am):
            res.append([0]*bn)
        for i in range(am):
            for j in range(an):
                if A[i][j]==0:
                    continue
                for k in range(bn):
                    res[i][j] += A[i][j]*B[j][k]
        return res
    
    @classmethod
    def findMinK(self , m , n , k ):
        # write code here
        i, j = 0, 0
        ml, nl = len(m), len(n)
        res = -1
        while i<ml and j<nl and k>0:
            if m[i] < n[j]:
                res = m[i]
                i += 1
            else:
                res = n[j]
                j += 1
            k -= 1
        if k>0:
            if i<ml:
                res = m[i+k-1]
            if j<nl:
                res = n[j+k-1]
        return res
a = [[1,0,0],[-1,0,5]]
b = [[6,0,0],[0,0,0],[0,0,1]]
Solution.abs_sort([-9,-3,0,1,2,8,10,14,18])
Solution.dotMultiply(a, b)
Solution.findMinK([1,2,2,7],[3,4,5],5)