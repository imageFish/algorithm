from copy import deepcopy
import torch
from tqdm import tqdm

class myHeap:
    def __init__(self, arr, max_heap=False):
        self.arr = deepcopy(arr)
        self.max_heap = max_heap
        self.len = len(arr)
        self.heapfiy()

    def heapfiy(self):
        l = self.len >> 1
        for i in reversed(range(l)):
            self.sift_up(i)

    def pop(self):
        if self.len == 0:
            return None
        res = self.arr[0]
        if self.len != 1:
            last_one = self.arr[-1]
            self.arr[0] = last_one
            self.arr.pop()
            self.len -= 1
            self.sift_up(0)
        else:
            self.arr.pop()
            self.len -= 1
        return res
    
    def append(self, item):
        self.arr.append(item)
        self.len += 1
        self.sift_down(self.len-1)

    def sift_up(self, st):
        while True:
            left_child = 2 * st + 1
            if left_child >= self.len:
                break
            right_child = 2 * st + 2
            if right_child < self.len:
                if (self.max_heap and self.arr[left_child] < self.arr[right_child]) or (not self.max_heap and self.arr[left_child] > self.arr[right_child]):
                    left_child = right_child
            if (self.max_heap and self.arr[left_child] < self.arr[st]) or (not self.max_heap and self.arr[left_child] > self.arr[st]):
                break
            else:
                self.arr[left_child], self.arr[st] = self.arr[st], self.arr[left_child]
                st = left_child
        
    def sift_down(self, pos):
        while True:
            parent_node = (pos-1) >> 1
            if parent_node < 0:
                break
            if self.max_heap:
                if self.arr[pos] > self.arr[parent_node]:
                    self.arr[pos], self.arr[parent_node] = self.arr[parent_node], self.arr[pos]
                else:
                    break
            else:
                if self.arr[pos] < self.arr[parent_node]:
                    self.arr[pos], self.arr[parent_node] = self.arr[parent_node], self.arr[pos]
                else:
                    break
def check(arr, max):
    for i in range(len(arr)):
        for j in [1, 2]:
            child = 2*i + j
            if child >= len(arr):
                continue
            if max and arr[i] < arr[child]:
                assert False
            elif not max and arr[i] > arr[child]:
                assert False
def test():
    for i in tqdm(range(1000)):
        arr_t = torch.randint(-3000, 3000, [1000])
        arr = arr_t.tolist()
        gt = torch.sort(arr_t)[0]

        heap_max = myHeap(arr, True)
        check(heap_max.arr, True)
        arr_sorted = []
        popped = heap_max.pop()
        while popped != None:
            arr_sorted.append(popped)
            popped = heap_max.pop()
        arr_sorted = arr_sorted[::-1]
        heap_max_t = torch.tensor(arr_sorted)
        mask = heap_max_t == gt
        if torch.sum(~mask) != 0:
            print('error')
            break
        
        heap_min = myHeap(arr)
        check(heap_max.arr, False)
        arr_sorted = []
        popped = heap_min.pop()
        while popped != None:
            arr_sorted.append(popped)
            popped = heap_min.pop()
        heap_max_t = torch.tensor(arr_sorted)
        mask = heap_max_t == gt
        if torch.sum(~mask) != 0:
            print('error')
            break
    print('success')
test()