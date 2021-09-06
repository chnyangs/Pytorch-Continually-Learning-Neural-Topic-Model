from copy import deepcopy

import torch
import torch.nn as nn
import torch.autograd as ag


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs):
        return self.linear(inputs).pow(2)


# Higher order derivative
def derivative_by_n(f, wrt, n):
    for i in range(n):
        grads = ag.grad(f, wrt, create_graph=True)[0]
        f = grads.sum()
    return grads


# Jacobian matrix
def jacob_matrix(f, x, outputs):
    x_size = list(x.size())
    x = x.unsqueeze(0).repeat(outputs, *([1] * (len(x_size)))).detach().requires_grad_(True)
    y = f(x)
    y.backward(torch.eye(outputs))
    return x.grad.view(outputs, *x_size)


# Hessian Matrix
def hessian_matrix(f, x):
    def get_grad(inputs):
        y = f(inputs)
        grad, = torch.autograd.grad(y, inputs, create_graph=True, grad_outputs=torch.ones_like(y))
        return grad

    x_size = x.numel()
    return jacob_matrix(get_grad, x, x_size)


def hessian_vector_product(model, inputs, vector):
    y = model(inputs)
    grads = torch.autograd.grad(y, model.parameters(), create_graph=True)
    prod = sum([(g * v).sum() for g, v in zip(grads, vector)])
    prod.backward()
    return [param.grad.detach() for param in model.parameters()]


def hessian_vector_product_diff(model, inputs, vectors, r=1e-7):
    def get_grad(model, x):
        y = model(x)
        grads = torch.autograd.grad(y, model.parameters())
        return grads

    def add(model, vector, r):
        model = deepcopy(model)
        for param, v in zip(model.parameters(), vector):
            param.data = param.data + r * v
        return model

    model_a = add(model, vectors, r)
    model_b = add(model, vectors, -r)
    grad_left_vectors = get_grad(model_a, inputs)
    grad_right_vectors = get_grad(model_b, inputs)
    return [(grad_left_vector - grad_right_vector) / (2 * r) for grad_left_vector, grad_right_vector
            in zip(grad_left_vectors, grad_right_vectors)]


if __name__ == "__main__":
    weight = torch.rand(3, 2)


    def fun(x):
        return (x @ weight).pow(2).sum(-1)


    x = torch.tensor([1, 2, 3.], requires_grad=True)
    print(hessian_matrix(fun, x))

    model = Model()
    x = torch.Tensor([1, 2])
    vectors = [torch.Tensor([[1, 5]]), torch.Tensor([2., 4])]
    print(vectors)
    print(hessian_vector_product(model, x, vectors))
