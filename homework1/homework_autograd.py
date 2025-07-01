import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simple_gradients():
    """
    Вычисляет градиенты функции f(x, y, z) = x² + y² + z² + 2xyz
    """
    x = torch.tensor(1.0, requires_grad=True, device=device)
    y = torch.tensor(2.0, requires_grad=True, device=device)
    z = torch.tensor(3.0, requires_grad=True, device=device)

    f = x**2 + y**2 + z**2 + 2 * x * y * z
    f.backward()

    # Аналитические градиенты:
    # df/dx = 2x + 2yz
    # df/dy = 2y + 2xz
    # df/dz = 2z + 2xy

    grad_analytic = {
        'df/dx': 2 * x.item() + 2 * y.item() * z.item(),
        'df/dy': 2 * y.item() + 2 * x.item() * z.item(),
        'df/dz': 2 * z.item() + 2 * x.item() * y.item(),
    }

    grad_autograd = {
        'df/dx': x.grad.item(),
        'df/dy': y.grad.item(),
        'df/dz': z.grad.item(),
    }

    return grad_autograd, grad_analytic


def mse_gradients():
    """
    Реализует MSE и находит градиенты по w и b
    """
    # Пример данных
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y_true = torch.tensor([2.0, 4.0, 6.0], device=device)

    w = torch.tensor(0.0, requires_grad=True, device=device)
    b = torch.tensor(0.0, requires_grad=True, device=device)

    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()
    loss.backward()

    return {
        'dL/dw': w.grad.item(),
        'dL/db': b.grad.item(),
        'loss': loss.item()
    }


def chain_rule_gradient():
    """
    Вычисляет градиент f(x) = sin(x^2 + 1)
    """
    x = torch.tensor(2.0, requires_grad=True, device=device)

    f = torch.sin(x**2 + 1)
    f.backward()

    # Аналитически df/dx = cos(x^2 + 1) * 2x
    analytic_grad = torch.cos(x**2 + 1) * 2 * x

    return {
        'grad_autograd': x.grad.item(),
        'grad_analytic': analytic_grad.item()
    }


def main():
    print("=== 2.1 Простые вычисления ===")
    auto, analytic = simple_gradients()
    print(f"Автоград: {auto}")
    print(f"Аналитика: {analytic}\n")

    print("=== 2.2 Градиенты MSE ===")
    mse_result = mse_gradients()
    print(mse_result, "\n")

    print("=== 2.3 Цепное правило ===")
    chain_result = chain_rule_gradient()
    print(chain_result)


if __name__ == "__main__":
    main()
