import collections
import os
import numpy as np
import scipy as sp

import bertini
import sdp_in

def start_points_optimum(C, A, b):
    # reshape feasible set
    ys = [np.random.choice((-1,1))*np.random.random() for i in range(m)]
    Ss = C - sum([ys[i]*A[i] for i in range(m)], np.zeros((n,n)))

    # choose y*_{m+1} to make S* strictly positive definite
    smallest_eig = min(np.linalg.eigvals(Ss))
    if smallest_eig <= 0:
        ymp1 = 1 - smallest_eig
        Amp1 = ymp1*np.eye(n)
    else:
        ymp1 = 0
        Amp1 = np.zeros((n,n))

    # compute S*, X*, b*
    Ss += Amp1
    Xs = np.linalg.inv(Ss)
    bs = [(A[i]*Xs).sum() for i in range(m)]

    return Xs,ys,bs

def start_points_feasible(C, A, b):
    bs = [0]*len(b) + [-1]
    return bs

def compute_optimum(C, A, b):
    Xs,ys,bs = find_start_points_optimum(C, A, b)
    # register constants with bertini
    constants = collections.OrderedDict()
    for i in range(m):
        for j in range(n):
            for k in range(j,n):
                constants["A{0}_{1}_{2}".format(i,j,k)] = A[i][j,k]
    for j in range(n):
        for k in range(j,n):
            constants["C{0}_{1}".format(j,k)] = C[j,k]
    for i in range(m):
        constants["b{0}".format(i)] = b[i]
        constants["bs{0}".format(i)] = bs[i]

    constants["y{0}".format(m)] = ymp1

    yvariables = []
    Xvariables = []

    # form base homotopy
    subfunctions = collections.OrderedDict()
    for j in range(n):
        for k in range(j,n):
            Xvariables.append("X{0}_{1}".format(j,k))
            eqterms = ["C{0}_{1}".format(j,k)]
            for i in range(m):
                eqterms.append("A{0}_{1}_{2}*y{0}".format(i,j,k))
            g = " - ".join(eqterms)
            if j == k:
                g += " + y{0}*mu".format(m)
            subfunctions["S{0}_{1}".format(j,k)] = g

    functions = collections.OrderedDict()

    for i in range(m):
        eqterms = []
        yvariables.append("y{0}".format(i))
        for j in range(n):
            eqterms.append("A{0}_{1}_{1}*X{1}_{1}".format(i,j))
            for k in range(j+1,n):
                eqterms.append("2*A{0}_{1}_{2}*X{1}_{2}".format(i,j,k))
        functions["f{0}".format(i)] = " + ".join(eqterms) + " - b{0}*(1 - mu) - bs{0}*mu".format(i)


    for i in range(n):
        for j in range(i,n):
            eqterms = []
            for k in range(n):
                if i <= k and k <= j:
                    eqterms.append("S{0}_{1}*X{1}_{2}".format(i,k,j))
                elif i <= k and k > j:
                    eqterms.append("S{0}_{1}*X{2}_{1}".format(i,k,j))
                elif i > k and k <= j:
                    eqterms.append("S{1}_{0}*X{1}_{2}".format(i,k,j))
                else:
                    eqterms.append("S{1}_{0}*X{2}_{1}".format(i,k,j))
            if i == j:
                h = " + ".join(eqterms) + " - mu"
            else:
                h = " + ".join(eqterms)
            functions["h{0}_{1}".format(i,j)] = h


    variable_group = Xvariables + yvariables
    # write out the input files
    dirname = "brun"
    try:
        os.mkdir(dirname)
    except OSError as e:
        if e.errno == 17: # file exists
            pass
        else:
            raise(e)

    Xpoints = []
    Spoints = []
    for i in range(n):
        for j in range(i, n):
            Xpoints.append(Xs[i,j])
            Spoints.append(Ss[i,j])

    startpoints = [bertini.Point(Xpoints + Spoints + ys)]
    bertini.write_bertini_input_file(dirname, variable_group, constants, subfunctions, functions, parameters={'mu':'t'}, pathvariables=['t'], options={"UserHomotopy":1})
    bertini.write_bertini_start_file(dirname, startpoints)
    bertini.run_bertini(dirname, "feasible_solve")

    try:
        optimum = bertini.read_solutions_file(dirname, "nonsingular_solutions")[0] # only one of these
    except ZeroDivisionError:
        optimum = bertini.read_solutions_file(dirname, "singular_solutions")[0] # only one of these

    Xcoords = optimum.coordinates[:int(n*(n+1)/2)]
#Scoords = optimum.coordinates[int(n*(n+1)/2):n*(n+1)]
    ycoords = optimum.coordinates[-m:]

    X = np.matrix(np.zeros((n,n)))
    S = np.matrix(np.zeros((n,n)))
    k = 0
    y = np.array([z.real for z in ycoords])
    for i in range(n):
        for j in range(i,n):
            X[i,j] = X[j,i] = Xcoords[k].real
            S[i,j] = S[j,i] = C[i,j] - sum([y[l]*A[l][i,j] for l in range(m)])
            k += 1

    return X,y,S

def compute_feasibility(C, A, b):
    # max -y_{m+1} s.t. C - \sum Ai*yi + I*y_{m+1} >= 0
    # => b~ = [0;...;0;-1] \in \R^{m+1}
C,A,b = sdp_in.read_in_SDP()
m = len(A)
n = C.shape[0] # square

X,y,S = compute_optimum(C,A,b)
