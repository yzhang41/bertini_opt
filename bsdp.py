"""
Question: 1. path tracking of solus to homogopy, all real solu?
          2. do we need "Gamma trick" technique?
             -- I think we need to put a random complex as the coeff. of start eqns.???
             -- it requires to modify the homotopy
"""
import collections
import os
import numpy as np
import scipy as sp

import bertini
import sdp_in

def find_start_points_optimum(C, A, b):
    # reshape feasible set
    m = len(b)
    n = C.shape[0] # square
    ys = [np.random.choice((-1,1))*np.random.random() for i in range(m)]
    print("ys = ", ys)
    # can we make ys to be random complex?
    # ys =[]
    # for i in range(m):
    # 	ran1 = np.random.choice((-1,1))*np.random.random()
    # 	ran2 = np.random.choice((-1,1))*np.random.random()
    # 	ran = complex(ran1, ran2)
    # 	print(ran)
    # 	ys.append(ran)
    Ss = C - sum([ys[i]*A[i] for i in range(m)], np.zeros((n,n)))

    # choose y*_{m+1} to make S* strictly positive definite
    smallest_eig = min(np.linalg.eigvals(Ss))
    print('smallest_eig = ', smallest_eig)
    if smallest_eig <= 0:
        ymp1 = smallest_eig - 1
        Amp1 = np.eye(n)
    else:
        ymp1 = 0
        Amp1 = np.zeros((n,n))

    # compute S*, X*, b*
    print("ymp1 = ", ymp1)
    print("Ss = ", Ss)
    Ss -= ymp1*Amp1 # make it to be "-"
    print("Ss =", Ss)
    Xs = np.linalg.inv(Ss)
    # A[i]*Xs: elem-wise mult. for np.ndarray
    bs = [(A[i]*Xs).sum() for i in range(m)]
    print("Xs =", Xs)
    print("bs = ", bs)

    return Xs,ys,bs,ymp1

# def find_start_points_feasible(C, A, b):
#     b = [0]*len(b) + [1]
#     return b

def compute_optimum(C, A, b):
	# compute start points
    # C, A are np.2darray (full dense matrix), b is a list of float
    # Xs: np.ndarray, ys, bs: list, ymp1 = scaler (nonpositive!)
    print('')
    print(type(C))
    print(C)
    for i in range(len(A)):
    	print(type(A[i]))
    	print(A[i])
    print(type(b))
    print(b)

    Xs,ys,bs,ymp1 = find_start_points_optimum(C, A, b)
    m = len(b)
    n = C.shape[0] # square

    # register constants with bertini
    constants = collections.OrderedDict()
    for i in range(m): # only input upper tri. of Ai
        for j in range(n):
            for k in range(j,n):
                constants["A{0}_{1}_{2}".format(i,j,k)] = A[i][j,k]
    for j in range(n): # only input upper tri. of C
        for k in range(j,n):
            constants["C{0}_{1}".format(j,k)] = C[j,k]
    for i in range(m): # bs stands for b*
        constants["b{0}".format(i)] = b[i]
        constants["bs{0}".format(i)] = bs[i]

    constants["y{0}".format(m)] = ymp1

    # two sets of variables: dim(y_var)=m, dim(X_var)=(n+1)*n/2
    yvariables = []
    Xvariables = []

    # form base homotopy
    # subfunction: S = C - \sum_i A_i * y_i - ysmp1 * \mu * Id, upper tri.
    # append X variables, upper tri.
    # here parameters={'mu':'t'}
    subfunctions = collections.OrderedDict()
    for j in range(n):
        for k in range(j,n):
            Xvariables.append("X{0}_{1}".format(j,k))
            eqterms = ["C{0}_{1}".format(j,k)]
            for i in range(m):
                eqterms.append("A{0}_{1}_{2}*y{0}".format(i,j,k))
            g = " - ".join(eqterms)
            if j == k:
                g += " - y{0}*mu".format(m) # make it to be "-"
            subfunctions["S{0}_{1}".format(j,k)] = g

    # two set of eqns
    functions = collections.OrderedDict()

    # first set (m eqns): A_i \cdot X - b_i * (1-mu) - bs_i * mu
    # append y variables
    for i in range(m):
        eqterms = []
        yvariables.append("y{0}".format(i))
        for j in range(n):
            eqterms.append("A{0}_{1}_{1}*X{1}_{1}".format(i,j))
            for k in range(j+1,n):
                eqterms.append("2*A{0}_{1}_{2}*X{1}_{2}".format(i,j,k))
        functions["f{0}".format(i)] = " + ".join(eqterms) + " - b{0}*(1 - mu) - bs{0}*mu".format(i)

    # second set ((n+1)*n/2 eqns): S.dot(X) - \mu * Id
    # only upper tri. part of S and X
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
    #Spoints = [] ## Questions: do we need Spoints to start?
    for i in range(n):
        for j in range(i, n):
            Xpoints.append(Xs[i,j])
            #Spoints.append(Ss[i,j])

    #startpoints = [bertini.Point(Xpoints + Spoints + ys)]
    startpoints = [bertini.Point(Xpoints + ys)]
    bertini.write_bertini_input_file(dirname, variable_group, constants, subfunctions, functions, parameters={'mu':'t'}, pathvariables=['t'], options={"UserHomotopy":1})
    bertini.write_bertini_start_file(dirname, startpoints)
    bertini.run_bertini(dirname, "optimum") # optimum mode

    try:
        optimum = bertini.read_solutions_file(dirname, "nonsingular_solutions")[0] # only one of these
    except ZeroDivisionError:
        optimum = bertini.read_solutions_file(dirname, "singular_solutions")[0] # only one of these

    Xcoords = optimum.coordinates[:int(n*(n+1)/2)]
    #Scoords = optimum.coordinates[int(n*(n+1)/2):n*(n+1)]
    ycoords = optimum.coordinates[-m:]

    ## add warning if X, S and y are "not" real!
    X = np.matrix(np.zeros((n,n)))
    S = np.matrix(np.zeros((n,n)))
    k = 0
    y = np.array([z.real for z in ycoords])
    for i in range(n):
        for j in range(i,n):
            X[i,j] = X[j,i] = Xcoords[k].real
            S[i,j] = S[j,i] = C[i,j] - sum([y[l]*A[l][i,j] for l in range(m)])
            k += 1

    return X,y,S # return np.matrix, np.array

def compute_feasibility(C, A, b):
    # max y_{m+1} s.t. C - \sum Ai*yi - I*y_{m+1} >= 0
    # => b~ = [0;...;0;1] \in \R^{m+1}
    ## enlarge b and A, A[m+1] = I
    #m = len(A)
    n = C.shape[0] # square
    # modify b, add one more dimension
    b = [0]*len(b) + [1] # make it to be "+"
    # modify A, add one more dimension
    A.append(np.eye(n))

    print('')
    print(type(C))
    print(C)
    for i in range(len(A)):
    	print(type(A[i]))
    	print(A[i])
    print(type(b))
    print(b)

    X,y,S = compute_optimum(C,A,b)

    return X,y,S

def postprocess(X,y,S,C,A,b):
    """
    Postprocessing stage:
     1. check whether the solu. is feasible
     2. check duality gap
     3. provide info. for feasibility test
     4. fix: C,A[i]: ndarray, b: list
    """
    print('\n--- Postprocessing: for small problem ---\n')
    tol = 1.0e-14 ## change here if necessary
    m = len(b)
    n = C.shape[0] # square
    # convert list b to np.array, C, A[i] np.ndarry -> np.matrix
    C = np.matrix(C)
    for i in range(m):
        A[i] = np.matrix(A[i])
    b = np.asarray(b)

    print(type(C))
    print(C)
    for i in range(len(A)):
    	print(type(A[i]))
    	print(A[i])
    print(type(b))
    print(b)

    # X feasible?
    smallest_eig_X = min(np.linalg.eigvals(X))
    print('smallest_eig_X = {0}'.format(smallest_eig_X))
    if smallest_eig_X >= -tol:
        print('X is spd and hence primal feasible!')
    else:
        print('X is NOT spd!')

    # trace(A_i .* X) - b_i = 0?
    test_equality_AXb = max([abs((np.multiply(A[i],X)).sum() - b[i]) for i in range(m)])
    if test_equality_AXb <= tol:
        print('trace(Ai * X) - bi ~= 0 for all i')
    else:
        print('trace(Ai * X) - bi ~= 0 for some i')

    # S*X = 0?
    test_equality_SX = (S*X).max()
    if test_equality_SX <= tol:
        print('S * X = 0')
    else:
        print('S * X ~= 0')

    # S feasible?
    smallest_eig_S = min(np.linalg.eigvals(S))
    print('\nsmallest_eig_S = {0}'.format(smallest_eig_S))
    if smallest_eig_S >= -tol:
        print('S is spd and hence dual feasible!')
    else:
        print('S is NOT spd!')

    # duality gap: C.*X - np.dot(b,y)
    # C*X: matrix multiplication, np.multiply(C,X): elem-wise mult.
    primal_obj_value = (np.multiply(C,X)).sum()
    print('\nprimal objective value = {0}'.format(primal_obj_value))
    dual_obj_value = np.dot(b, y)
    print('dual objective value = {0}'.format(dual_obj_value))
    print('duality gap = {0}'.format(primal_obj_value - dual_obj_value))
    print('')


# main function
if __name__ == '__main__':
    
    # read data from file located in examples: C, A, b
    cwd = os.getcwd()
    example_tag = '6' ## change here, also can be input on command line
    example_dirname = os.path.join(cwd, 'examples')
    mode = 2

    C,A,b = sdp_in.read_in_SDP(example_dirname, example_tag)

    # task 1: optimum solve
    if mode == 1:
		X,y,S = compute_optimum(C, A, b)
	    # postprocessing
	    # print(type(C))
	    # print(C)
	    # for i in range(len(A)):
	    # 	print(type(A[i]))
	    # 	print(A[i])
	    # print(type(b))
	    # print(b)
		postprocess(X,y,S,C,A[:],b) # pass a copy of A: A[:]
	    # print(type(C))
	    # print(C)
	    # for i in range(len(A)):
	    # 	print(type(A[i]))
	    # 	print(A[i])
	    # print(type(b))
	    # print(b)

    # task 2: feasibility test ## failed path! ???
    if mode == 2:
		print(type(C))
		print(C)
		for i in range(len(A)):
			print(type(A[i]))
			print(A[i])
		print(type(b))
		print(b)
		X,y,S = compute_feasibility(C, A, b)
		print('')
		print(type(C))
		print(C)
		for i in range(len(A)):
			print(type(A[i]))
			print(A[i])
		print(type(b))
		print(b)

		last_idx = len(b)
		eps = 1.0e-10
		print("max y_m+1:", y[last_idx])
		if ( y[last_idx] > eps):
			print("Dual of SDP is strict feasible!")
		elif (y[last_idx] < -eps):
			print("Dual of SDP is infeasible!")
		else: #==0?
			print("Two cases for Dual of SDP:\n feasible, but not strict feasible \n OR infeasible")
