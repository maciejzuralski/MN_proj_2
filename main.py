# Maciej Żuralski 193367
import copy
import math
import matplotlib.pyplot as plt
import time


# Special function for exercise A
def special_function(x):
    # f + 1 = 3 + 1 = 4
    return math.sin(x * 4)


# Calculate matrix Multiplication
def matrix_multiplication(matrix_1, matrix_2):
    mat1_high = len(matrix_1)
    mat1_width = len(matrix_1[0])
    mat2_width = len(matrix_2[0])

    new_matrix = [[0 for _ in range(mat2_width)] for _ in range(mat1_high)]

    for y in range(mat1_high):
        for x in range(mat2_width):
            new_matrix[y][x] = sum(matrix_1[y][i] * matrix_2[i][x] for i in range(mat1_width))

    return new_matrix


# Calculate matrix subtraction
def matrix_subtraction(matrix_1, matrix_2):
    mat_high = len(matrix_1)
    mat_width = len(matrix_1[0])

    new_matrix = [[0 for _ in range(mat_width)] for _ in range(mat_high)]

    for y in range(mat_high):
        for x in range(mat_width):
            new_matrix[y][x] = matrix_1[y][x] - matrix_2[y][x]

    return new_matrix


# Calculate matrix addition
def matrix_addition(matrix_1, matrix_2):
    mat_high = len(matrix_1)
    mat_width = len(matrix_1[0])

    new_matrix = [[0 for _ in range(mat_width)] for _ in range(mat_high)]

    for y in range(mat_high):
        for x in range(mat_width):
            new_matrix[y][x] = matrix_1[y][x] + matrix_2[y][x]

    return new_matrix


# Exercise A - create matrix equation
def create_matrix_equation(matrix_size, fun, a1, a2, a3):
    a_matrix_new = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    b_matrix_new = [[0] for _ in range(matrix_size)]
    x_matrix_new = [[0] for _ in range(matrix_size)]

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

    return a_matrix_new, x_matrix_new, b_matrix_new


# Calculate vector residuum
def vector_residuum_norm(matrix_a, matrix_x, matrix_b):
    residiuum = matrix_subtraction(matrix_multiplication(matrix_a, matrix_x), matrix_b)
    norm = 0

    for y in range(len(residiuum)):
        for x in range(len(residiuum[0])):
            norm += residiuum[y][x] * residiuum[y][x]

    return math.sqrt(norm)


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


# Diagonal matrix inversion
def diagonal_matrix_inversion(matrix):
    new_matrix = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        new_matrix[i][i] = 1/matrix[i][i]

    return new_matrix


# Matrix sigh change
def matrix_sigh_change(matrix):
    new_matrix = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    for y in range(len(matrix)):
        for x in range(len(matrix)):
            new_matrix[y][x] = matrix[y][x] * -1

    return new_matrix


# Jacobi
def jacobi(A, x, b, max_error):
    residuum = 100
    residuum_arr = []
    iteration = 0
    iterations = []

    L, U, D = split_matrix_to_L_U_D(A)
    D_inv = diagonal_matrix_inversion(D)
    L_plus_U = matrix_addition(L, U)

    while max_error < residuum and iteration < 100:
        x = matrix_multiplication(D_inv, matrix_subtraction(b, matrix_multiplication(L_plus_U, x)))
        #x = matrix_addition(matrix_multiplication(M, x), Bm)
        residuum = vector_residuum_norm(A, x, b)
        residuum_arr.append(residuum)
        iteration += 1
        iterations.append(iteration)

    return x, residuum_arr, iterations


def matrix_forward_substitution(L, b):
    size = len(b)
    x = [[0] for _ in range(size)]

    for row in range(size):
        x[row][0] = b[row][0]
        for col in range(row):
            x[row][0] -= L[row][col] * x[col][0]
        x[row][0] /= L[row][row]

    return x


# Matrix forward substitution
def my_matrix_forward_substitution(L, b):
    size = len(b)
    x = [[0] for _ in range(size)]

    x[0][0] = b[0][0] / L[0][0]
    for row in range(1, size):
        suma = sum(L[row][col] * x[col][0] for col in range(row))
        x[row][0] = (b[row][0] - suma) / L[row][row]

    return x


# Matrix backward substitution
def my_matrix_backward_substitution(U, b):
    size = len(b)
    x = [[0] for _ in range(size)]

    x[size - 1][0] = b[size - 1][0] / U[size - 1][size - 1]
    for row in range(size - 2, -1, -1):
        suma = sum(U[row][col] * x[col][0] for col in range(row + 1, size))
        x[row][0] = (b[row][0] - suma) / U[row][row]

    return x


# Gauss
def gauss(A, x, b, max_error):
    residuum = 100
    residuum_arr = []
    iteration = 0
    iterations = []

    L, U, D = split_matrix_to_L_U_D(A)
    L_plus_D = matrix_addition(L, D)

    while max_error < residuum and iteration < 100:
        x = matrix_forward_substitution(L_plus_D, matrix_subtraction(b, matrix_multiplication(U, x)))
        residuum = vector_residuum_norm(A, x, b)
        residuum_arr.append(residuum)
        iteration += 1
        iterations.append(iteration)

    return x, residuum_arr, iterations


# create lu for factorization
def create_l_u(A):
    n = len(A)
    U = copy.deepcopy(A)
    L = [[0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1

    for i in range(1, n):
        for j in range(i):
            L[i][j] = U[i][j] / U[j][j]
            for k in range(n):
                U[i][k] -= L[i][j] * U[j][k]

    return L, U


# exercise B
def ex_B():
    n = 967
    a1 = 8  # 5 + 3 = 8
    a2 = -1
    a3 = -1
    A, x, b = create_matrix_equation(n, special_function, a1, a2, a3)
    start = time.time()
    jacobi_x, jacobi_err, jacobi_iter = jacobi(A, x, b, 1e-9)
    end = time.time()
    print("Jacobi:", end - start)
    A, x, b = create_matrix_equation(n, special_function, a1, a2, a3)
    start = time.time()
    gauss_x, gauss_err, gauss_iter = gauss(A, x, b, 1e-9)
    end = time.time()
    print("Gauss:", end - start)

    plt.figure()
    plt.plot(jacobi_iter,jacobi_err)
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('error norm')
    plt.title('Jacobi method')
    plt.show()

    plt.figure()
    plt.plot(gauss_iter, gauss_err)
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('error norm')
    plt.title('Gauss method')
    plt.show()


# exercise C
def ex_C():
    n = 967
    a1 = 3
    a2 = -1
    a3 = -1
    A, x, b = create_matrix_equation(n, special_function, a1, a2, a3)
    jacobi_x, jacobi_err, jacobi_iter = jacobi(A, x, b, 1e-9)
    A, x, b = create_matrix_equation(n, special_function, a1, a2, a3)
    gauss_x, gauss_err, gauss_iter = gauss(A, x, b, 1e-9)

    print("J min:", min(jacobi_err))
    print("G min:", min(gauss_err))

    plt.figure()
    plt.plot(jacobi_iter, jacobi_err)
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('error norm')
    plt.title('Jacobi method')
    plt.show()

    plt.figure()
    plt.plot(gauss_iter, gauss_err)
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('error norm')
    plt.title('Gauss method')
    plt.show()


# exercise D
def ex_D():
    n = 967
    a1 = 3
    a2 = -1
    a3 = -1
    A, x, b = create_matrix_equation(n, special_function, a1, a2, a3)
    start = time.time()
    L, U = create_l_u(A)
    x = my_matrix_backward_substitution(U, my_matrix_forward_substitution(L, b))
    end = time.time()
    print("LU factorization:", end - start)
    res = vector_residuum_norm(A, x, b)
    print(res)


# exercise E
def ex_E():
    Ns = [100, 500, 1000, 3000]
    for n in Ns:
        print("For n =", n, "time in second")
        a1 = 8  # 5 + 3 = 8
        a2 = -1
        a3 = -1
        A, x, b = create_matrix_equation(n, special_function, a1, a2, a3)
        start = time.time()
        jacobi_x, jacobi_err, jacobi_iter = jacobi(A, x, b, 1e-9)
        end = time.time()
        print("Jacobi:", end - start)
        A, x, b = create_matrix_equation(n, special_function, a1, a2, a3)
        start = time.time()
        gauss_x, gauss_err, gauss_iter = gauss(A, x, b, 1e-9)
        end = time.time()
        print("Gauss:", end - start)
        #A, x, b = create_matrix_equation(n, special_function, a1, a2, a3)
        #start = time.time()
        #L, U = create_l_u(A)
        #x = my_matrix_backward_substitution(U, my_matrix_forward_substitution(L, b))
        #end = time.time()
        #print("LU factorization:", end - start)


if __name__ == '__main__':
    # ex_B()
    # ex_C()
    # ex_D()
    # ex_E()
