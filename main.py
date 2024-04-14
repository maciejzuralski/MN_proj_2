# Maciej Żuralski 193367
import math
import numpy as np
import matplotlib.pyplot as plt


# Special function for exercise A
def special_function(x):
    # f + 1 = 3 + 1 = 4
    return math.sin(x * 4)


# Exercise A - create matrix equation
def create_matrix_equation(matrix_size, fun):
    a1 = 8  # 5 + 3 = 8
    a2 = -1
    a3 = -1

    a_matrix_new = np.zeros((matrix_size, matrix_size))
    b_matrix_new = np.zeros((matrix_size, 1))

    for i in range(matrix_size - 2):
        a_matrix_new[i][i] = a1
        a_matrix_new[i + 1][i] = a2
        a_matrix_new[i][i + 1] = a2
        a_matrix_new[i + 2][i] = a3
        a_matrix_new[i][i + 2] = a3

        b_matrix_new[i][0] = fun(i)

    a_matrix_new[matrix_size - 2][matrix_size - 2] = a1
    a_matrix_new[matrix_size - 1][matrix_size - 1] = a1
    a_matrix_new[matrix_size - 2][matrix_size - 1] = a2
    a_matrix_new[matrix_size - 1][matrix_size - 2] = a2

    b_matrix_new[matrix_size - 2][0] = fun(matrix_size - 2)
    b_matrix_new[matrix_size - 1][0] = fun(matrix_size - 1)

    return a_matrix_new, b_matrix_new


if __name__ == '__main__':
    n = 967
    a_matrix, b_matrix = create_matrix_equation(n, special_function)
