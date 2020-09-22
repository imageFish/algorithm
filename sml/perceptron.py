import torch
x = torch.tensor([
    [3,3],
    [4,3],
    [1,1.0],
])
y = torch.tensor([
    [1.0], [1], [-1]
])
def _perceptron():
    w = torch.tensor([[0],[0]])
    b = torch.tensor([0])

    while True:
        wx = torch.mm(x, w)
        loss = -y*(wx+b)
        loss_flag = loss >= 0
        if torch.any(loss_flag):
            for i in range(x.size(0)):
                if loss_flag[i]:
                    w = w + (y[i]*x[i]).unsqueeze(-1)
                    b = b + y[i]
                    print(w, b)
                    break
        else:
            break
def _com_perceptron():
    gram = torch.mm(x, x.transpose(-1, -2))
    n = torch.zeros(3,1)
    b = torch.zeros(1)
    while True:
        loss = y * (torch.mm(gram, n*y) + b)
        loss_flag = loss <= 0
        if torch.any(loss_flag):
            for i in range(loss_flag.size(0)):
                if loss_flag[i]:
                    n[i][0] += 1
                    b += y[i][0]
                    print(n.tolist(), b.tolist())
                    break
        else:
            break
x = [
        [2,3], [5,4], [9, 6], [4,7], [8,1], [7,2]
]
n = 2
cnt = 0
def kdTree(l, r, d):
    if l==r:
        print(x[r])
        return
    if l>r: return
    d = d%n
    x[l:r+1] = sorted(x[l:r+1], key=lambda x: x[d])
    m = (l+r+1) >> 1 
    print(x[m])
    kdTree(l, m-1, d+1)
    kdTree(m+1, r, d+1)
def kdTreeSearch(l, r, pi):
    if l<r: return False, False
    if l==r: return l, ((3-x[pi][0])**2 + (4.5-x[pi][1])**2)**0.5
    d = d%n
    m = (l+r+1) >> 1 
    if pt
    return pi
pt = (3, 4.5)
kdTree(0, len(x)-1, 0)
kdTreeSearch(0, len(x)-1, 1)