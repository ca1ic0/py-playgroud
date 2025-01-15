  # 创建一个生成器
from collections.abc import Iterable
from collections.abc import Iterator

def fib():
    n, a, b = 0, 0, 1
    while True:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'

print(isinstance(fib(),Iterable))
print(isinstance(fib(),Iterator))

print(isinstance(fib, Iterator))
print(isinstance(fib, Iterable))