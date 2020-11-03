## minumun number of a any-order array
```python
# Given an order of the array, a and b is one of the array sperately, a is on the left of b
if a+b > b+a: # string concat
    # exchanging a and b will make the order less
    # so the problem is equal to that sort the array by comparing the string consisted of each two number
```

## xor
&, |, ^, meets the law of commutation and association.

## ordered array sum
right-top of the leading diagonal (secondary diagonal) is column and row ordered, but not all sum elements

## Josephus problem 
```python
# m%n is the start idx s_i after removing the m-th numbert of the [0, ```, n-1]
# s_i = ((m-1)%n+1)%n
#     = (m-1)%n + 1%n = (m-1+1)%n = m %n

# f is the remaing number of problem (n-1, m)
# so for problem (n, m), remaining idx = (m%n+f)%n = (m+f)%n
f = 0
for i in range(2, n+1):
    f = (m+f)%i
return f
```

## logical operation, and if 
```python
# equal expression
if condition:
    then

condition && then

# bit sheft instead, no int multiplication
s = a * b

s = 0
while b != 0:
    s += (b&1) * a
    a << 1
    b >> 1

# bit opts replace add, multi:
def add(self, a, b):
    x = 0xffffffff # there is no bit numbers in python saving INT, 
    a, b = a & x, b & x # &x is get the comlitary code
    while b!=0:
        a, b = a^b, (a&b)<<1 & x 
    return a if a <= 0x7fffffff else ~(a ^ x) # max positive int 0x7fffffff, if a number is greater then its, it is nagetive and need to ~ its bit flag.
```
## LRU
least recently used最近最少使用，使用队列维护存放的值：
- 存key时，如果key在队列中，直接修改。如果不在，在队列尾端添加key，如果满了删除队首
- 查询时，如果有，则返回值，没有则-1
## 二项分布
$$
E = np\\
D = np(1-p)
$$