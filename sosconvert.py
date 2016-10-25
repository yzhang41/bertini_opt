#!/usr/bin/python3

from sympy.polys.orderings import monomial_key
import sympy
import os

def symmetric_indeterminate_matrix(n):
    Q = sympy.zeros(n)
    for i in range(n):
        Q[i,i] = 'q{0}_{0}'.format(i)
        for j in range(i,n):
            Q[i,j] = Q[j,i] = 'q{0}_{1}'.format(i, j)

    return Q

def sdp_equiv(poly, vargroup):
    degree = poly.total_degree()
    mons = sympy.Matrix(sorted(sympy.itermonomials(vargroup, degree//2), key=monomial_key('grlex', reversed(vargroup))))

    n = len(mons)
    Q = symmetric_indeterminate_matrix(n)

    ind = (mons.T*Q*mons)[0,0].as_poly(vargroup)

    n = Q.shape[0]
    all_q = [] # sorted
    for i in range(n):
        all_q.append('q{0}_{0}'.format(i))
        for j in range(i+1, n):
            all_q.append('q{0}_{1}'.format(i, j))

    all_q_sym = sympy.symbols(all_q)

    eqs = {}

    for mon in ind.monoms():
        ind_coeff = ind.coeff_monomial(mon)
        poly_coeff = poly.coeff_monomial(mon)
        
        if ind_coeff in eqs.keys():
            print("duplicate! " + str(ind_coeff))

        eqs[ind_coeff] = poly_coeff

    As = []
    bs = []
    for lhs in eqs.keys():
        bs.append(eqs[lhs])
        lhsp = lhs.as_poly(all_q_sym)
        A = sympy.zeros(n)
        for monom in lhsp.monoms():
            coeff = lhsp.coeff_monomial(monom)
            index = next((i for i, x in enumerate(monom) if x), None)
            symbol = all_q[index][1:]
            ij = symbol.split("_")
            i = int(ij[0])
            j = int(ij[1])
            if i == j:
                A[i,j] = coeff
            else:
                A[i,j] = A[j,i] = 0.5*coeff
        As.append(A)

    return As, bs
            
############### MAIN
from sympy.abc import w,x,y,z
#### THIS EXAMPLE (example 10) WORKS FAST (n = 10 ==> 55 variables) 
#vargroup = [x,y,z]
#q = 10*x**2*y**2 + 2*x**2*y*z + 4*x**2*y + 2*x**2*z**2 - 6*x**2*z + 7*x**2 - 6*x*y**2*z - 8*x*y**2 - 24*x*y*z**2 + 16*x*y*z - 2*x*y - 8*x*z**2 + 6*x*z - 2*x + y**4 - 6*y**3*z + 19*y**2*z**2 + 2*y**2*z - y**2 + 6*y*z**3 + 12*y*z**2 + 9*z**4 + z**2 - 2*z + 2 

#### THIS EXAMPLE (example 11) TAKES A LITTLE BIT MORE TIME (n = 15 ==> 120 variables)
#vargroup = [w, x, y, z]
#q = 4*w**4 - 8*w**3*y + 12*w**3 + 18*w**2*x**2 - 6*w**2*x + 12*w**2*y**2 - 12*w**2*y + 4*w**2*z**2 + 11*w**2 + 18*w*x**3 + 18*w*x**2*y + 6*w*x**2*z + 10*w*x*y*z - 2*w*x*y - 6*w*x*z**2 + 16*w*x*z + 10*w*x - 8*w*y**3 + 12*w*y**2 - 12*w*y*z - 6*w*y - 6*w*z**2 - 8*w + 9*x**4 + 6*x**3 + 14*x**2*y**2 + 12*x**2*y*z - 20*x**2*y - 5*x**2*z**2 + 16*x**2*z - 3*x**2 + 12*x*y**2 - 12*x*y*z**2 + 8*x*y*z - 8*x*y - 8*x*z**2 + 10*x*z - 10*x + 4*y**4 + y**2*z**2 - 6*y**2*z + 27*y**2 - 2*y*z**3 + 12*y*z**2 - 20*y*z + 18*y + 10*z**4 - 6*z**3 + 23*z**2 - 6*z + 9

#### THIS EXAMPLE (example 12) TAKES A LONG TIME (n = 20 ==> 210 variables)
vargroup = [x, y, z]
q = 11*x**6 - 2*x**5*y + 6*x**5 + 18*x**4*y**2 - 20*x**4*y*z + 4*x**4*y + 24*x**4*z**2 + 20*x**4*z - 4*x**4 - 4*x**3*y**3 + 28*x**3*y**2*z + 8*x**3*y**2 - 2*x**3*y*z**2 + 10*x**3*y*z + 32*x**3*y + 22*x**3*z**3 + 12*x**3*z - 8*x**3 + 9*x**2*y**4 + 6*x**2*y**3*z + 31*x**2*y**2*z**2 + 12*x**2*y**2*z + 44*x**2*y**2 - 30*x**2*y*z**3 - 24*x**2*y*z**2 + 26*x**2*y*z - 32*x**2*y + 4*x**2*z**4 - 12*x**2*z**3 + 21*x**2*z**2 + 6*x**2*z + 18*x**2 + 12*x*y**5 + 22*x*y**4*z + 14*x*y**4 - 4*x*y**3*z**2 + 8*x*y**3*z + 16*x*y**3 + 28*x*y**2*z**3 - 44*x*y**2 - 30*x*y*z**4 - 10*x*y*z**3 + 30*x*y*z**2 + 26*x*y*z + 14*x*y + 12*x*z**5 + 8*x*z**4 + 24*x*z**3 + 36*x*z**2 - 20*x*z + 6*x + 11*y**6 + 6*y**5*z + 16*y**5 + 10*y**4*z**2 + 6*y**4*z - 12*y**4 + 6*y**3*z**3 - 26*y**3*z**2 + 6*y**3*z - 16*y**3 + 31*y**2*z**4 - 12*y**2*z**3 + 6*y**2*z**2 + 12*y**2*z + 24*y**2 + 6*y*z**5 + 20*y*z**4 + 8*y*z**3 + 22*y*z**2 - 6*y*z + 9*z**6 + 12*z**5 + 9*z**4 - 18*z**3 + 18*z**2 + 1

poly = q.as_poly(vargroup)

As, bs = sdp_equiv(poly, vargroup)

m = len(As)
n = As[0].shape[0]

dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
tags = [int(f.lstrip("A").strip(".txt")) for f in os.listdir(dirname) if f.startswith("A")]

current_tag = max(tags) + 1

with open(os.path.join(dirname, "A{0}.txt".format(current_tag)), "w") as fh:
    print(m, file=fh)
    for A in As:
        print(n, file=fh)
        for i in range(n):
            print(float(A[i,i]), file=fh)
            for j in range(i+1, n):
                print(float(A[i,j]), file=fh)

with open(os.path.join(dirname, "b{0}.txt".format(current_tag)), "w") as fh:
    for b in bs:
        print(float(b), file=fh)

with open(os.path.join(dirname, "C{0}.txt".format(current_tag)), "w") as fh:
    print(n, file=fh)
    for i in range(n*(n+1)//2):
        print(0.0, file=fh)
