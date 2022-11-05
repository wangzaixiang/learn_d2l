import random
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # [1024, 2]
    y = torch.matmul(X, w) + b  # [1024]
    y += torch.normal(0, 0.01, y.shape)
    y = y.reshape(-1, 1)    # [1024, 1]
    return X, y

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))

    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            if param.grad != None:
                param -= lr * param.grad / batch_size
                param.grad.zero_()


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    features, labels = synthetic_data(true_w, true_b, 1024)

    batch_size = 16
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break

    print('features:', features[0], '\nlabel:', labels[0])

    lr = 0.1
    num_epochs = 20
    net = linreg
    loss = squared_loss
    # w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    # b = torch.zeros(1, requires_grad=True)
    # b = torch.tensor([4.19]) # no grad here
    w = torch.rand(2, 1, requires_grad=True)
    b = torch.rand(1, requires_grad=True)

    print(f"w = {w}, b = {b}")

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels): # X: [16, 2], Y: [16 1]
            y_hat = net(X, w, b)  # y_hat: [16, 1]
            l = loss(y_hat, y)    # l: [16, 1]
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss( net(features, w, b), labels)
            print(f'epoch {epoch} loss:{float(train_l.mean()):f}')

    print(f"w = {w}, b = {b}")

    pass
