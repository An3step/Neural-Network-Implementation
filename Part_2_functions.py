from Part_1_functions import *

def assert_mulShape(X1: np.ndarray, X2: np.ndarray):
    assert X1.shape[1] == X2.shape[0], f'X1 should have shape {X2.shape[0]} while it has {X1.shape[1]}'

def backward_multiple_add(x: np.ndarray, y: np.ndarray, sigma: Array_Function) -> float:
    a = x + y
    dsda = deriv(sigma, a)
    dadx, dady = 1, 1
    return dsda * dadx, dsda * dady

def matmul_forward(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    return N

def matmul_backward(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    
    dYdX = np.transpose(W)
    
    return dYdX

def matrix_forward_extra(X: np.ndarray, W: np.ndarray, sigma: Array_Function) -> np.ndarray:
    
    assert X.shape[1] == W.shape[0], f'X should have shape {W.shape[0]} while it has {X.shape[1]}'
    
    N = X @ W
    print(N)
    S = sigma(N)
    
    return S

def matrix_function_backward(X: np.ndarray, W: np.ndarray, sigma: Array_Function) -> np.ndarray:
    assert_mulShape(X, W)
    N = X @ W
    dS_dN = deriv(sigma, N)
    dNu_dX = matmul_backward(X, W)
    return dS_dN * dNu_dX