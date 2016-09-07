import numpy as np

def map(f, x):
    return [f(y) for y in x]

def striplines(filename):
    with open(filename, 'r') as fh:
        lines = [l.strip() for l in fh.readlines() if l != '\n']
    return lines

def read_in_symmetric_matrix(lines, n):
    M = np.zeros((n,n), dtype=float)
    vals = map(float, lines)

    k = 0
    for i in range(n):
        for j in range(i,n):
            x = vals[k]
            if i == j:
                M[i,i] = x
            else:
                M[i,j] = M[j,i] = x
            k += 1

    return M

def read_in_C_matrix(filename):
    lines = striplines(filename)
    n = int(lines[0]) # one dimension of (square) C matrix

    C = read_in_symmetric_matrix(lines[1:], n)
    return C

def read_in_A_matrices(filename):
    A = []

    lines = striplines(filename)
    m = int(lines[0]) # number of A matrices
    lines = lines[1:]

    Acount = int(len(lines)/m) # number of unique entries in Ai, plus 1

    for i in range(m):
        Alines = lines[:Acount]
        lines = lines[Acount:]
        n = int(Alines[0])
        Ai = read_in_symmetric_matrix(Alines[1:], n)
        A.append(Ai)

    return A

def read_in_b_vector(filename):
    lines = striplines(filename)
    b = [float(l) for l in lines]

    return b

def read_in_SDP():
    Cfile = 'C.txt'
    Afile = 'A.txt'
    bfile = 'b.txt'

    C = read_in_C_matrix(Cfile)
    A = read_in_A_matrices(Afile)
    b = read_in_b_vector(bfile)

    return C,A,b
