import collections
import os
import numpy as np
import scipy as sp

import bertini
import sdp_in

C,A,b = sdp_in.read_in_SDP()
m = len(A)
n = C.shape[0] # square

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

# form base homotopy
subfunctions = collections.OrderedDict()
for i in range(n):
    for j in range(i+1, n):
        subfunctions["X{0}_{1}".format(j,i)] = "X{0}_{1}".format(i,j)
        subfunctions["S{0}_{1}".format(j,i)] = "S{0}_{1}".format(i,j)

functions = collections.OrderedDict()

yvariables = []
Xvariables = []
Svariables = []

for i in range(m):
    eqterms = []
    yvariables.append("y{0}".format(i))
    for j in range(n):
        eqterms.append("A{0}_{1}_{1}*X{1}_{1}".format(i,j))
        for k in range(j+1,n):
            eqterms.append("2*A{0}_{1}_{2}*X{1}_{2}".format(i,j,k))
    functions["f{0}".format(i)] = " + ".join(eqterms) + " - b{0}*(1 - mu) - bs{0}*mu".format(i)

for j in range(n):
    for k in range(j,n):
        Xvariables.append("X{0}_{1}".format(j,k))
        Svariables.append("S{0}_{1}".format(j,k))
        eqterms = "C{0}_{1},S{0}_{1}".format(j,k).split(',')
        for i in range(m):
            eqterms.append("A{0}_{1}_{2}*y{0}".format(i,j,k))
        g = " - ".join(eqterms)
        if j == k:
            g += " + y{0}*mu".format(m)
        functions["g{0}_{1}".format(j,k)] = g

for i in range(n):
    for j in range(i,n):
        eqterms = []
        for k in range(n):
            eqterms.append("S{0}_{1}*X{1}_{2}".format(i,k,j))
        if i == j:
            h = " + ".join(eqterms) + " - mu"
        else:
            h = " + ".join(eqterms)
        functions["h{0}_{1}".format(i,j)] = h

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

variable_group = Xvariables + Svariables + yvariables
bertini.write_bertini_input_file(dirname, variable_group, constants, subfunctions, functions, parameters={'mu':'t'}, pathvariables=['t'], options={"UserHomotopy":1})
bertini.write_bertini_start_file(dirname, startpoints)
bertini.run_bertini(dirname, "feasible_solve")

# recover feasible point
try:
    feasible_point = bertini.read_solutions_file(dirname, "nonsingular_solutions")[0] # only one of these
except ZeroDivisionError:
    feasible_point = bertini.read_solutions_file(dirname, "singular_solutions")[0] # only one of these

Xcoords = feasible_point.coordinates[:int(n*(n+1)/2)]
ycoords = feasible_point.coordinates[-m:]

primal_feas = np.matrix(np.zeros((n,n)))
k = 0
for i in range(n):
    for j in range(i,n):
        primal_feas[i,j] = primal_feas[j,i] = Xcoords[k].real
        k += 1
dual_feas = np.array([z.real for z in ycoords])

# max -y_{m+1} s.t. C - \sum Ai*yi + I*y_{m+1} >= 0
# => b~ = [0;...;0;-1] \in \R^{m+1}
