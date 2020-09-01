from copy import deepcopy
cnt = 0


def selection_sort(arr):
    l = len(arr)
    for i in range(l-1):
        m = arr[i]
        idx = i
        for j in range(i+1, l):
            if m > arr[j]:
                m = arr[j]
                idx = j
        arr[i], arr[idx] = arr[idx], arr[i]
    return arr

def bubble_sort(arr):
    l = len(arr)
    for i in range(l-1):
        for j in range(i, l-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# def shell_sort(arr): #采用排序对 两两排序  间隔l/2->1

def quick_sort(arr, l=None, r=None):
    global cnt
    if l == None:
        l, r = 0, len(arr) - 1
    i, j = l, r
    if l >= r:
        return
    
    key = arr[l]
    while i < j:
        while i < j and arr[j] > key:
            j -= 1
        if i < j:
            arr[i] = arr[j]
            i += 1

        while i < j and arr[i] <= key:
            i += 1
        if i < j:
            arr[j] = arr[i]
            j -= 1
    arr[i] = key
    quick_sort(arr, 0, i-1)
    quick_sort(arr, i+1, r)

def merge_sort(arr, l, r):
    if l >= r:
        return
    m = int((r-l)/2 + l)
    merge_sort(arr, l, m) 
    merge_sort(arr, m+1, r)
    i, j = l, m+1
    
    merged = []
    while i <= m and j <= r:
        if arr[i] < arr[j]:
            merged.append(arr[i])
            i += 1
        else:
            merged.append(arr[j])
            j += 1
    if i <= m:
        merged.extend(arr[i:m+1])
    if j <= r:
        merged.extend(arr[j:r+1])
    arr[l:r+1] = merged

def heap_sort(arr):
    import heapq
    heapq.heapify(arr)
    res = []
    while arr:
        res.append(heapq.heappop(arr))
    return res

def creat_heap(arr):
    a = deepcopy(arr)
    # for i in range

    
        




a = [15, 23, 14, 28, 13, 17, 42, 20]
b = deepcopy(a)
func = [heap_sort]
for fun in func:
    res = fun(a)
    print(res)
    print(a)
b.sort()
assert a == b
print('ok')