import torch

# Автоматически определяем устройство (gpu или cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_tensors():
    """
    Создает несколько тензоров с заданными параметрами.
    """
    tensor_random = torch.rand((3, 4), device=device)
    tensor_zeros = torch.zeros((2, 3, 4), device=device)
    tensor_ones = torch.ones((5, 5), device=device)
    tensor_range = torch.arange(16, device=device).reshape((4, 4))

    return tensor_random, tensor_zeros, tensor_ones, tensor_range


def tensor_operations(A: torch.Tensor, B: torch.Tensor):
    """
    Выполняет различные операции с тензорами A и B.
    """
    assert A.shape == (3, 4) and B.shape == (4, 3), "Неверные размерности тензоров A и B"

    A_t = A.T
    matmul_result = torch.matmul(A, B)
    elementwise_mul = A * B.T
    total_sum = A.sum()

    return A_t, matmul_result, elementwise_mul, total_sum


def indexing_slicing():
    """
    Демонстрирует индексацию и срезы тензора 5x5x5.
    """
    tensor = torch.arange(125, device=device).reshape((5, 5, 5))

    first_row = tensor[0]                      # первая строка это плоскость 5 на 5, берем первую плоскость
    last_column = tensor[:, :, -1]            # последний столбец
    center_slice = tensor[2:4, 2:4, 2:4]       # центральная подматрица 2x2x2
    even_indices = tensor[::2, ::2, ::2]       # элементы с чётными индексами

    return first_row, last_column, center_slice, even_indices


def reshape_tensor():
    """
    Создает тензор из 24 элементов и преобразует его в различные формы.
    """
    tensor = torch.arange(24, device=device)

    shapes = {
        "2x12": tensor.reshape((2, 12)),
        "3x8": tensor.reshape((3, 8)),
        "4x6": tensor.reshape((4, 6)),
        "2x3x4": tensor.reshape((2, 3, 4)),
        "2x2x2x3": tensor.reshape((2, 2, 2, 3))
    }

    return shapes


def main():
    print("=== 1.1 Создание тензоров ===")
    tensors = create_tensors()
    for i, t in enumerate(tensors, 1):
        print(f"Тензор {i}:\n{t}\n")

    print("=== 1.2 Операции с тензорами ===")
    A = torch.rand((3, 4), device=device)
    B = torch.rand((4, 3), device=device)
    A_t, matmul_result, elementwise_mul, total_sum = tensor_operations(A, B)
    print(f"Транспонированный A:\n{A_t}")
    print(f"А @ B:\n{matmul_result}")
    print(f"A * B.T:\n{elementwise_mul}")
    print(f"Сумма элементов A: {total_sum}\n")

    print("=== 1.3 Индексация и срезы ===")
    row, col, center, even = indexing_slicing()
    print(f"Первая строка:\n{row}")
    print(f"Последний столбец:\n{col}")
    print(f"Центральная подматрица 2x2x2:\n{center}")
    print(f"Элементы с чётными индексами:\n{even}\n")

    print("=== 1.4 Преобразование форм ===")
    reshaped = reshape_tensor()
    for name, tensor in reshaped.items():
        print(f"Форма {name}:\n{tensor}\n")


if __name__ == "__main__":
    main()
