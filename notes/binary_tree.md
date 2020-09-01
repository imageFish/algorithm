## Sequence 
| pre-order | mid-order | post-order |
| :-------: | :-------: | :--------: |
|  h->l->r  |  l->h->r  |  l->r->h   |

## Binary Search Tree
Rebuild tree by post-orders.
1. flip post-orders. init root to $+\infty$
2. $root$ is for left child tree, and every node of left child tree is less then root. For right child tree, there is no root, and everr node is greater than $root^*$.
```python
def verifyPostorder(self, postorder):
    stack, root = [], float('+inf')
    for n in postorder[::-1]:
        if n > root:
            return False
        while stack and stack[-1] > n:
            root = stack.pop()
        stack.append(n)
    return True
```

## Rebuild tree from pre-order with post-order or mid-order
To rebuild a binary tree, the pre-order is necessary, including post-order or mid-order.
### prove

## Complete Binary Tree
For saving space, use $List$ to store $CBT$. For parent node $N_i$,
$$
\begin{aligned}
    \text{left child} = 2i+1 \\
    \text{right child} = 2i+2
\end{aligned}
$$
for node $N_i$,
$$
\text{parent node} = (i - 1) \models 2
$$
### Prove
$$
\begin{aligned}
    \text{for layer i, j-th node} &= 2^i+j\\
    \text{left node} &= 2^{i+1} + 2*j + 1\\
    &=2\times(2^i+j) + 1 \\
    \text{right node} &= 2^{i+1} + 2*j + 2\\
    &=2\times(2^i+j) + 2
\end{aligned}
$$
## Balance Tree