import torch
x = torch.tensor([
    [3,3],
    [4,3],
    [1,1],
])
y = torch.tensor([
    [1], [1], [-1]
])
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