# -*- coding: UTF-8 -*-
"""
Remark: 1. three modes:
            mode 1: optimum_solve
            mode 2: feasibility_dual (does not output feasible pt for P) ???
            mode 3: feasibility_primal (does not output feasible pt for D) ???
            mode 4: test
        
        python -m cProfile -o output bsdp.py
        python -m pstats output
        ### 11/09/2016 ###
"""
import collections
import os
import numpy as np
import scipy as sp

import bertini
import sdp_in

# for convenience
def print_data(C, A, b):
    print('')
    print(type(C))
    print(C)
    for i in range(len(A)):
        print(type(A[i]))
        print(A[i])
        print(type(b))
        print(b)

def test_linear_depend(A, *args):
    """
    test linear dependence between A_i and possible *args
    use: numpy.linalg.matrix_rank(M, tol=None)
    """
    m = len(A)
    n = A[0].shape[0] # square
    M = np.zeros((m,int(n*(n+1)/2))) # row-major data for efficiency?
    for i in range(m):
        M[i,:] = A[i][np.triu_indices(n)]
    for item in args: # args is a tuple
        M = np.vstack([M, item[np.triu_indices(n)]])
        m += 1
    rank = np.linalg.matrix_rank(M)
    r = min(m, int(n*(n+1)/2))

    return rank < r # if this True, then linear dependent

def test_feasibility_equality(A,b):
    """
    test feasibility of A_i \cdot X = b_i, i = 1,...,m
    Since A_i are already linearly independent, it's not necessary to solve them directly!
    """
    m = len(A)
    n = A[0].shape[0] # square
    N = int(n*(n+1)/2)
    feasible = False
    if m<=N:
        print("m <= n(n+1)/2, underdetermined or square system for equality constraints")
        print("Since A_i are already linearly independent")
        print("A_i \cdot X = b_i, i = 1,...,m is always feasible!")
        print("Continue...")
    elif m>N:
        print("Warning: m > n(n+1)/2, overdetermined system for equality constraints")
        print("Warning: waiting to be tested!")
        M = np.zeros((m,N))
        for i in range(m):
            idx = 0
            for j in range(n):
                for k in range(j,n):
                    if j==k:
                        M[i,idx] = A[i][j,k]
                    else:
                        M[i,idx] = 2*A[i][j,k]
                    idx += 1
            if idx != N-1:
                print("idx error!")
                exit()
        x_lstsq = np.linalg.lstsq(A,b)[0]
        residual = np.linalg.norm(np.dot(M,x_lstsq) - b)
        print("residual = {0}".format(residual))
        tol = 1.0e-8
        if abs(residual) > tol:
            print("residual too big, exit")
            exit()
        # Q,R = linalg.qr(A) # qr decomposition of A
        # Qb = dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
        # x_qr = linalg.solve(R,Qb) # solving R*x = Q^T*b

def find_start_points_optimum(C, A, b, mode):
    # reshape feasible set
    print('---Inside find_start_points_optimum---')
    m = len(b)
    n = C.shape[0] # square
    ys = [np.random.choice((-1,1))*np.random.random() for i in range(m)] # len(ys) == m
    if mode == 2:
    	ys[m-1] = 1 # the last one is modified to be 1, M = 2
    #print("ys = ", ys)
    Ss = C - sum([ys[i]*A[i] for i in range(m)], np.zeros((n,n))) # why need np.zeros((n,n))?

    # choose \tilde{y}:=ysmp1 to make S* strictly positive definite
    smallest_eig = min(np.linalg.eigvals(Ss))
    #print('smallest_eig = ', smallest_eig)
    if smallest_eig <= 0:
        ysmp1 = smallest_eig - 1
        Amp1 = np.eye(n)
    else:
        ysmp1 = 0
        Amp1 = np.zeros((n,n))
    if mode == 2: # feasibility_dual mode
        if (ys[m-1]+ysmp1) > 0: # ys[m-1] is origional ys_{m+1}
            print('ys[m-1]+ysmp1 = {0} > 0 and hence SDP-D is strictly feasible'.format(ys[m-1]+ysmp1))
            print('Early exit in ' + find_start_points_optimum.__name__)
            print(ys)
            #print('Still continue, ok! If only do the SDP-D feasibility test, we can exit here!')
            exit()

    # compute S*, X*, b*
    Ss -= ysmp1*Amp1 # make it to be "-"
    Xs = np.linalg.inv(Ss) # in mode 3, Xs is  actually \tilde{Xs}

    # if mode == 3:
        # trace_Ai_zero = True
        # for i in range(m):
        #     if np.trace(A[i]) != 0:
        #         trace_Ai_zero = False
        #         break
        # if trace_Ai_zero:
        #     print("All trace(A_i) = 0, thus dual of (SDP-P)_2 is infeasible")
        #     print("inf. of (SDP-P)_2 is negative INF, thus (SDP-P) is strict feasible!")
        #     exit()

    bs = [(A[i]*Xs).sum() for i in range(m)]
    if mode == 2: # modify bs[m-1]
    	# beta_s = 1 since it is (M - ys[m-1])^{-1}, M = 2 and ys[m-1] = 1
    	bs[m-1] = np.trace(Xs) + 1 # it seems that we do not need this info.
    if mode == 3: # len(ys) = m+1 now since ys[m] = \lambda
        # enlarge ys for mode 3 (feasibility_primal)
        #ys.append(np.random.choice((-1,1))*np.random.random())
        # Set M = 2, choose gamma_s=1>0, then lambda_s = -1
        ys.append(-1)
    	# now len(ys) = m+1, the last variable represents lambda
        bs = [bs[i] - ys[m]*np.trace(A[i]) for i in range(m)]
    trace_Ss = np.trace(Ss)

    # ysmp1 is \tilde{y}
    if mode == 3:
        return Xs,ys,bs,ysmp1,trace_Ss # for mode 3, len(ys) = m+1 instead of m
    else:
        return Xs,ys,bs,ysmp1

def compute_optimum(C, A, b, mode):
    # compute start points
    # C, A are np.2darray (full dense matrix), b is a list of float
    # Xs: np.ndarray, ys, bs: list, ysmp1 = scaler (nonpositive!)
    # print_data(C,A,b)
    print("---Inside compute_optimum---")
    if mode == 3:
        Xs,ys,bs,ysmp1,trace_Ss = find_start_points_optimum(C, A, b, mode)
    else: # mode == 1 or 2
        Xs,ys,bs,ysmp1 = find_start_points_optimum(C, A, b, mode)
    m = len(b) # use this, b/c in mode 3, len(y)=m+1 (adding \lambda to the last position of y)
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

    #constants["y{0}".format(m)] = ymp1
    constants["ysmp1"] = ysmp1 # use ysmp1:=\tilde{y} bc y_m is reserved for \lambda in mode 3

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
                g += " - ysmp1*mu" # make it to be "- ysmp1 * mu"
            subfunctions["S{0}_{1}".format(j,k)] = g
    #del eqterms # need this?

    # two set of eqns
    functions = collections.OrderedDict()

    # first set (m eqns): A_i \cdot X - b_i * (1-mu) - bs_i * mu
    # append y variables
    if mode == 1:
    	mm = m
    if mode == 2:
    	mm = m-1 # cut off the last one
    for i in range(mm):
        eqterms = []
        yvariables.append("y{0}".format(i))
        for j in range(n):
            eqterms.append("A{0}_{1}_{1}*X{1}_{1}".format(i,j))
            for k in range(j+1,n):
                eqterms.append("2*A{0}_{1}_{2}*X{1}_{2}".format(i,j,k))
        functions["f{0}".format(i)] = " + ".join(eqterms) + " - b{0}*(1 - mu) - bs{0}*mu".format(i)
    if mode == 2: # add back the m position of the equality constraint
    	yvariables.append("y{0}".format(m-1))
    	eqterms = []
    	for i in range(n):
    		eqterms.append("X{0}_{0}".format(i))
    	# M = 2
    	functions["f{0}".format(m-1)] = "( 1 + mu * {0} - ".format(np.trace(Xs)) + "-".join(eqterms) + ") * (2-y{0}) - mu".format(m-1)

    # modify first set of eqns. for mode 3
    if mode == 3:
    	yvariables.append("y{0}".format(m)) # add variable lambda to yvariables y[m]
        for i in range(m):
        #     eqterms = []
        #     for j in range(n):
        #         eqterms.append("y{0} * A{1}_{2}_{2}".format(m,i,j))
        #     add_terms = " - " + " - ".join(eqterms)
        #     functions["f{0}".format(i)] += add_terms
        	functions["f{0}".format(i)] += " - y{0} * {1}".format(m,np.trace(A[i]))

    # for mode 3, add one more eqn to f[m]: XXXXXX S \cdot I - 1*(1-\mu) - \mu * trace(Ss) = 0
    # New: (1 + \mu trace(Ss) - trace(S)) * (\lambda + M) - \mu = 0
    if mode == 3:
        eqterms = [] # waster a lot of old eqterms??? modify later for efficiency
        for i in range(n):
            eqterms.append("S{0}_{0}".format(i))
        #print(eqterms)
        #["f{0}".format(m)] = " + ".join(eqterms) + " - (1 - mu) - mu * {0}".format(trace_Ss) 
        # " - 1 - mu * {0}".format(trace_Ss-1.0)
        functions["f{0}".format(m)] = "(" + "1 + mu * {0}".format(trace_Ss) + " - " + " - ".join(eqterms) + ")" + "* (y{0} + 2) - mu".format(m)

    # second set ((n+1)*n/2 eqns): S.dot(X) - \mu * Id
    # only upper tri. part of S and X
    # symmetric version!
    for i in range(n):
        eqterms = []
        for k in range(n):
            if i <= k:
                eqterms.append("2*S{0}_{1}*X{0}_{1}".format(i, k))
            else:
                eqterms.append("2*S{0}_{1}*X{0}_{1}".format(k, i))
        h = " + ".join(eqterms) + " - 2*mu"
        functions["h{0}_{0}".format(i)] = h

        for j in range(i+1, n):
            eqterms = []
            for k in range(n):
                if i <= k and k <= j:
                    eqterms.append("S{0}_{1}*X{1}_{2} + X{0}_{1}*S{1}_{2}".format(i,k,j))
                elif i <= k and k > j:
                    eqterms.append("S{0}_{1}*X{2}_{1} + X{0}_{1}*S{2}_{1}".format(i,k,j))
                elif i > k and k <= j:
                    eqterms.append("S{1}_{0}*X{1}_{2} + X{1}_{0}*S{1}_{2}".format(i,k,j))
                else:
                    eqterms.append("S{1}_{0}*X{2}_{1} + X{1}_{0}*S{2}_{1}".format(i,k,j))
                h = " + ".join(eqterms)
            functions["h{0}_{1}".format(i,j)] = h

    #variable_group = Xvariables + yvariables
    ## make variable_group to be list of list
    if mode == 3:
        variable_group = [Xvariables, yvariables[:len(yvariables)-1], [yvariables[len(yvariables)-1]] ]
    else:
        variable_group = [Xvariables, yvariables]

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
    bertini.write_bertini_input_file(dirname, variable_group, constants, subfunctions, functions, parameters={'mu':'t'}, pathvariables=['t'], options={"UserHomotopy":2, "SecurityLevel":1})
    bertini.write_bertini_start_file(dirname, startpoints)
    bertini.run_bertini(dirname, "optimum") # optimum mode

    try:
        optimum = bertini.read_solutions_file(dirname, "nonsingular_solutions")[0] # only one of these
    except ZeroDivisionError:
        optimum = bertini.read_solutions_file(dirname, "singular_solutions")[0] # only one of these

    Xcoords = optimum.coordinates[:int(n*(n+1)/2)]
    #Scoords = optimum.coordinates[int(n*(n+1)/2):n*(n+1)]
    ycoords = optimum.coordinates[int(n*(n+1)/2):]
    
    ## add warning if X, S and y are "not" real!
    #isComplex1 = [np.iscomplex(i) for i in Xcoords]
    #print("\nisComplex for Xvariables?:",isComplex1)
    #isComplex2 = [np.iscomplex(i) for i in ycoords]
    #print("isComplex for yvariables?:",isComplex2)
    # for i,j in zip(isComplex1, isComplex2):
    #     if i == True or j == True:
    #         print("Warning: non-real solution!")
    #         break
    tol = 1.0e-10 # change here if necessary
    Xcoords_imag = [z.imag for z in Xcoords]
    normX_imag = np.linalg.norm(Xcoords_imag)
    ycoords_imag = [z.imag for z in ycoords]
    normy_imag = np.linalg.norm(ycoords_imag)
    if normX_imag > tol:
        print("\nWarning: non-real solution for X!")
    if normy_imag > tol:
        print("\nWarning: non-real solution for y!") # if y is real, then S is of course real!
    if max(normX_imag,normy_imag) > tol:
        print("Output info. in postprocessing stage is incorrect!")
        print("Go to check solution files for more info!")    	

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
    # in mode 3, y[m] is lambda!

def compute_feasibility_dual(C, A, b, mode):
    # max y_{m+1} s.t. C - \sum Ai*yi - I*y_{m+1} >= 0
    # => b~ = [0;...;0;1] \in \R^{m+1}
    ## enlarge b and A, A[m+1] = I

    print('---Inside compute_feasibility_dual---')
    n = C.shape[0] # square
    # check whether I is a linear combination of A_i's
    # if so, we come to a conclusion: SDP-D is strictly feasible (can be proved straightforwardly!)
    # no need to continue computing in this case!
    # If C > 0, then it is strict feasible immediately (set y_i = 0 for all i)
    linear_depend_info = test_linear_depend(A, np.eye(n))
    if linear_depend_info:
        print('Identity matrix is a linear combination of A_i (i=1,...,m)')
        print('Hence SDP-D is strictly feasible!')
        print('Early exit in ' + compute_feasibility_dual.__name__)
        #print('----')
        exit()
    else:
        print('We can safely append A with I and modify b, continue...')
    print('---Continue---')

    # modify b, add one more dimension, but this change only affect inside comp_feasibiility_dual, why???
    b = [0]*len(b) + [1] # make it to be "+"
    # modify A, add one more dimension
    A.append(np.eye(n)) # this modification will affect outside this function if A (not A[:]) is passed in
    # print_data(C,A,b)
    X,y,S = compute_optimum(C,A,b,mode)

    return X,y,S # len(y) == m+1

def compute_feasibility_primal(C, A, b, mode):
    """
    min \lambda { X + \lambda I >= 0, A_i \cdot X = b_i, i=1,...,m }
    add one more primal variable \lambda, this new variable will append to the last of yvariables
    this will bring one more constraint eqn. into the KKT system: S \cdot I - 1 = 0
    Set C = 0
    Remark: note that here we use X to represent X + \lambda I, hence we need to modify the first set of eqns.
    """
    print('---Inside compute_feasibility_primal---')
    n = C.shape[0] # square
    C = np.zeros((n,n)) # or change in the homotopy section to remove C-info.
    # for convenience, set C = 0 to reuse the code

    X,y,S = compute_optimum(C,A,b,mode)

    return X,y,S # y[m] is lambda in mode 3!

def postprocess(X,y,S,C,A,b):
    """
    Postprocessing stage:
     1. check whether the solu. is feasible
     2. check duality gap
     3. provide info. for feasibility test
     4. fix: C,A[i]: ndarray, b: list
    """
    tol = 1.0e-14 ## change here if necessary
    m = len(b)
    n = C.shape[0] # square
    # convert list b to np.array, C, A[i] np.ndarry -> np.matrix
    C = np.matrix(C)
    for i in range(m):
        A[i] = np.matrix(A[i])
    b = np.asarray(b)
    # print_data(C,A,b)
    
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

# main function (for test purpose)
if __name__ == '__main__':
    ## set up parameters here
    ## cf. /examples/readme.txt for more details
    mode_dict = { '1':'optimum_solve', '2':'feasibility_test_dual', '3':'feasibility_test_primal', '4':'other test' }
    example_tag = '9' ## change here, also can be input on command line
    mode = 2

    print('---------------Example {0}-------------------'.format(example_tag))
    print('---------------mode {0}: {1}-------------------'.format(mode, mode_dict[str(mode)]))

    # read data from file located in examples: C, A, b
    cwd = os.getcwd()
    example_dirname = os.path.join(cwd, 'examples')
    C,A,b = sdp_in.read_in_SDP(example_dirname, example_tag)

    # task 0: first check linear dependence of A_i
    # this step can be skipped if A_i are already linearly indepenent
    # but there is no harm doing it for small test problem
    linear_depend_info = test_linear_depend(A)
    if linear_depend_info:
        print('Please remove redundacy between A_i (i=1,...,m) before running this code!')
        print('Early exit in ' + __name__)
        exit()

    # test feasibility of A_i \cdot X = b_i
    test_feasibility_equality(A,b)
    # if not feasible, print("A_i \cdot X = b_i, i = 1,...,m are infeasible")
    # print("(SDP-P) is infeasible!")
    # exit()
    # ##

    # task 1: optimum solve
    if mode == 1:
        X,y,S = compute_optimum(C, A, b, mode)
        # postprocessing
        # print_data(C,A,b)
        print('\n---------------Output: postprocessing: for small problem-------------------')
        postprocess(X,y,S,C,A[:],b) # pass a copy of A: A[:]
        print('X=',X)
        print('y=',y)
        # C,A,b will not change after this call
        # if the data A is large, just pass A as "reference"
        ## fix this later to make it more clear and efficient!

    # task 2: feasibility_dual
    if mode == 2:
        X,y,S = compute_feasibility_dual(C, A[:], b, mode) # len(b) == m
        ## output: len(y) == m+1
        print('\n---------------Output: feasibility Test for SDP_D-------------------')
        last_idx = len(b) # == len(y) - 1
        eps = 1.0e-10 # change here if necessary
        print("optimal dual obj. value: max y_m+1 = {0}".format(y[last_idx]))
        beta = 1 - np.trace(X)
        print("beta = {0}---should be nonnegative!".format(beta))
        optimal_primal_value = np.multiply(np.matrix(C),X).sum() + beta * 2 # M = 2
        print("optimal primal obj. value: C \cdot X + beta * M = {0}". format(optimal_primal_value))
        print("duality gap = {0}".format(optimal_primal_value - y[last_idx]))
        print("eps = {0}\n".format(eps))
        if ( y[last_idx] > eps):
            print("Dual of SDP is strict feasible!")
        elif (y[last_idx] < -eps):
            print("Dual of SDP is infeasible!")
        else: #==0?
            print("Two possible cases for Dual of SDP:\n feasible, but not strict feasible \n OR infeasible")
            print("(SDP-D): C - sum{yi*Ai} - eps*Id >= 0 is infeasible, C - sum{yi*Ai} + eps*Id >= 0 is feasible!")
        print("X = ")
        print(X)
        print("min_eig(X) = {0}".format(min(np.linalg.eigvals(X))))
        print("\nS = ")
        print(S)
        print("min_eig(S) = {0}".format(min(np.linalg.eigvals(S))))

    if mode == 3: # feasibility_primal
        X,y,S = compute_feasibility_primal(C, A[:], b, mode)
        print('\n---------------Output: feasibility Test for SDP_P-------------------')
        n = C.shape[0]
        m = len(b) # len(y) == m+1
        #print(y)
        lambd = y[m]
        gamma = 1 - np.trace(S)
        print("gamma = {0}---should be nonnegative!".format(gamma))
        optimal_dual_value = np.dot(b, y[:m]) - gamma*2 # M = 2
        print('opt_optim_value = {0}, opt_dual_value = {1}'.format(lambd, optimal_dual_value))
        print("duality gap = {0}".format(lambd - optimal_dual_value))
        eps = 1.0e-10
        print('eps = {0}'.format(eps))
        if lambd < -eps:
            print("Primal of SDP is strict feasible!")
        elif lambd > eps:
            print("Primal of SDP is infeasible!")
        else: # ==0?
            print("Two possible cases for Primal of SDP:\n feasible, but not strict feasible \n OR infeasible")
            print("(SDP-P): X + eps * I >= 0  is feasible, X - eps * I >= 0 is infeasible!\n")
        #print_data(X-y[m]*np.eye(n), A, b)
        #print(X)
        #print(S)
        ### THIS IS the real X we want in SDP-P!
        print("X=")
        print(X - y[m]*np.eye(n))
        print("eigenvlaue of X = {0}".format(min(np.linalg.eigvals(X - y[m]*np.eye(n)))))
        print("\nS=")
        print(S)
        #print(min(np.linalg.eigvals(S)))
        #print(np.trace(S))

    # task 4: other test
    if mode == 4:
        A = np.array([[1,2,3], [4,5,6], [7,8,9]])
        print(np.trace(A))
        print(type(A))
        print(A)
        A = A/(np.trace(A)/2)
        print(A)
        print(np.trace(A))
        a = A[np.triu_indices(3)]
        print(type(a))
        print(a)
        M = np.zeros((2,6))
        M[0,:] = a
        print(M)
        #A = numpy.vstack([A, newrow])
        M = np.vstack([M, a])
        print(M)

        y = np.array([0, 1, 2, 3, 4])
        print(y)
        print(y[-3:])
        print(y[:4])

        ll = [[1,2], [3,4]]
        print(len(ll))
        print(ll[0], ll[1])
        print(ll[1][:1])

    print('')
