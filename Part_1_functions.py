from typing import Callable 
import numpy as np

def deriv(func: Callable[[np.ndarray], np.ndarray], input_: np.ndarray, delta: float = 0.001) -> np.ndarray:
    '''
    Вычисление производной функции func в каждой точке массива input_
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

from typing import List

Array_Function = Callable[[np.ndarray], np.ndarray]

Chain = List[Array_Function]

def chain_length_2(chain: Chain, a: np.ndarray) -> np.ndarray:
    '''
    вычисляет подряд значение вложенной функции в точке a
    '''
    assert len(chain) == 2, 'в цепочке должно быть 2 функции'
    f1 = chain[0]
    f2 = chain[1]
    return f2(f1(a))

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def chain_deriv_2(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    assert len(chain) == 2, 'Длина chain должна быть 2'
    f1 = chain[0]
    f2 = chain[1]
    f1_x = f1(input_range)
    df1_x = deriv(f1, input_range)
    df2_f1_x = deriv(f2, f1_x)
    return df2_f1_x * df1_x

def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, x, 0.2 * x)

def square(x: np.ndarray) -> np.ndarray:
    return x ** 2

def identity(x: np.ndarray) -> np.ndarray:
    return x

def chain_deriv_3(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    assert len(chain) == 3, 'Длина цепочки должна быть 3'
    f1, f2, f3 = chain
    f1_x = f1(input_range)
    f2_x = f2(input_range)
    df1_x = deriv(f1, input_range)
    df2_f1 = deriv(f2, f1_x)
    df3_f2 = deriv(f3, f2_x)
    return df3_f2 * df2_f1 * df1_x

def chain_length_3(chain: Chain, a: np.ndarray) -> np.ndarray:
    '''
    вычисляет подряд значение вложенной функции в точке a
    '''
    assert len(chain) == 3, 'в цепочке должно быть 3 функции'
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    return f3(f2(f1(a)))