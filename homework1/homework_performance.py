import torch
import time
import pandas as pd

# Проверяем наличие cuda
has_cuda = torch.cuda.is_available()
device_gpu = torch.device("cuda" if has_cuda else "cpu")
device_cpu = torch.device("cpu")

# Готовим тензоры
shapes = [
    (64, 1024, 1024),
    (128, 512, 512),
    (256, 256, 256),
]

operations = [
    ("Мат. умножение", lambda a, b: torch.matmul(a, b.transpose(-1, -2))),
    ("Сложение", lambda a, b: a + b),
    ("Умножение", lambda a, b: a * b),
    ("Транспонирование", lambda a, _: a.transpose(-1, -2)),
    ("Суммирование", lambda a, _: a.sum()),
]


def measure_time_cpu(func, a, b):
    start = time.time()
    func(a, b)
    torch.cuda.synchronize() if has_cuda else None
    return (time.time() - start) * 1000  # в миллисекундах


def measure_time_gpu(func, a, b):
    if not has_cuda:
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def benchmark():
    results = []

    for shape in shapes:
        print(f"\nРазмер: {shape}")
        size_str = f"{shape[0]}x{shape[1]}x{shape[2]}"
        for op_name, op_func in operations:
            # Создание тензоров
            a_cpu = torch.rand(shape, device=device_cpu)
            b_cpu = torch.rand(shape, device=device_cpu)

            # CPU
            time_cpu = measure_time_cpu(op_func, a_cpu, b_cpu)

            # GPU
            if has_cuda:
                a_gpu = a_cpu.to(device_gpu)
                b_gpu = b_cpu.to(device_gpu)
                time_gpu = measure_time_gpu(op_func, a_gpu, b_gpu)
                speedup = round(time_cpu / time_gpu, 2) if time_gpu else "-"
            else:
                time_gpu = "-"
                speedup = "-"

            results.append({
                "Операция": op_name,
                "Размер": size_str,
                "CPU (мс)": round(time_cpu, 2),
                "GPU (мс)": round(time_gpu, 2) if time_gpu != "-" else "-",
                "Ускорение": f"{speedup}x" if speedup != "-" else "-"
            })

    df = pd.DataFrame(results)
    print("\nРезультаты:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    benchmark()
