## count number
Count m, where $0\le m \le 9$, apperance times of numbers between 1~n

For $n_i$, i-th number of number $n_ln_{l-1}\cdots n_0$, m apperance times is as following:
$$
\begin{aligned}
    high &= n_ln_{l-1}\cdots n_{i+1} \\
    low &= n_{i-1}\cdots n_0 \\
    digit &= 10^{i-1}\\  
\end{aligned}\\
times = \begin{cases}
    high*digit \quad \text{if}\quad n_i < m\\
    high*digit + low + 1 \quad \text{if}\quad n_i = m\\
    (high+1)*digit \quad \text{if} \quad n_i > m\\

\end{cases}
$$
```python
def countDigitOne(self, n):
    """
    :type n: int
    :rtype: int
    """
    high = n // 10
    low = 0
    digit = 1
    cur = n % 10
    res = 0
    while high != 0 or cur != 0:
        if cur == 0:
            res += high * digit
        elif cur == 1:
            res += high * digit + low + 1
        else:
            res += (high + 1) * digit
        digit *= 10
        cur = high % 10
        high = high // 10
        low = n % digit
    return res
```