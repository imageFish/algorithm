def reversePairs(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    # import bisect
    nl = len(nums)
    bt = [0]*(nl+1)
    def lowbit(i):
        return i&-i
    def update(i, k):
        while i<=nl:
            bt[i] += k
            i += lowbit(i)
    def get_sum(i):
        res = 0
        while i>0:
            res += bt[i]
            i -= lowbit(i)
        return res
    res = 0
    # tmp = sorted(nums)
    # for i in range(nl):
    #     nums[i] = bisect.bisect_left(tmp, nums[i])+1
    tmp = []
    for i in range(nl):
        tmp.append([nums[i], i])
    tmp.sort()
    i = 0
    while i < nl: # keep relative order
        j = i
        nums[tmp[j][1]] = j+1
        while i+1<nl and tmp[i][0] == tmp[i+1][0]:
            i += 1
            nums[tmp[i][1]] = j+1
        i += 1
    for i in range(nl-1, -1, -1):
        res += get_sum(nums[i]-1)
        update(nums[i], 1)
    return res
print(reversePairs([1,3,2,3,1]))