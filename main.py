# Maciej Żuralski 193367
import math
import numpy as np
import matplotlib.pyplot as plt


# Special function for exercise A
def special_function(x):
    # f + 1 = 3 + 1 = 4
    return math.sin(x * 4)


# Calculate matrix Multiplication
def matrix_multiplication(matrix_1, matrix_2):
    mat1_high = len(matrix_1)
    mat1_width = len(matrix_1[0])
    mat2_width = len(matrix_2[0])

    new_matrix = [[0 for _ in range(mat1_high)] for _ in range(mat2_width)]

    for y in range(mat1_high):
        for x in range(mat2_width):
            new_matrix[y][x] = sum(matrix_1[y][i] * matrix_2[i][x] for i in range(mat1_width))

    return new_matrix


# Calculate matrix subtraction
def matrix_subtraction(matrix_1, matrix_2):
    mat_high = len(matrix_1)
    mat_width = len(matrix_1[0])

    new_matrix = [[0 for _ in range(mat_high)] for _ in range(mat_width)]

    for y in range(mat_high):
        for x in range(mat_width):
            new_matrix[y][x] = matrix_1[y][x] - matrix_2[y][x]

    return new_matrix


# Exercise A - create matrix equation
def create_matrix_equation(matrix_size, fun):
    a1 = 8  # 5 + 3 = 8
    a2 = -1
    a3 = -1

    a_matrix_new = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    b_matrix_new = [[0] for _ in range(matrix_size)]

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


# Calculate vector residuum
def vector_residuum(matrix_a, matrix_x, matrix_b):
    return matrix_subtraction(matrix_multiplication(matrix_a, matrix_x), matrix_b)


# Split matrix to L, U, D
def split_matrix_to_L_U_D(matrix):
    width = len(matrix)

    L = [[0 for _ in range(width)] for _ in range(width)]
    U = [[0 for _ in range(width)] for _ in range(width)]
    D = [[0 for _ in range(width)] for _ in range(width)]

    for y in range(width):
        for x in range(width):
            if x < y:
                L[y][x] = matrix[y][x]
            elif y < x:
                U[y][x] = matrix[y][x]
            elif x == y:
                D[y][x] = matrix[y][x]

    return L, U, D


if __name__ == '__main__':
    n = 967
    a_matrix, b_matrix = create_matrix_equation(20, special_function)
    split_matrix_to_L_U_D(a_matrix)

