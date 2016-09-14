"""
Remark: 1. path tracking of solus to homogopy, all real solu?
        2. three modes:
        	mode 1: optimum_solve
        	mode 2: feasibility_dual
        	mode 3: feasibility_primal
        	mode 4: test
        3. next task: add feasibility_primal mode
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
		tempA = A[i]
		# print(tempA)
		# a = tempA[np.triu_indices(n)]
		# print(a)
		M[i,:] = tempA[np.triu_indices(n)]
	for item in args: # args is a tuple
		M = np.vstack([M, item[np.triu_indices(n)]])
		m += 1
	rank = np.linalg.matrix_rank(M)
	#print(rank, m)

	return rank < m # if this True, then linear dependent

## write a separate function for D-SDP feasibility test?
## if I = \sum A_i c_i, then ... we arrive a conclusion, no need to continue?
def find_start_points_optimum(C, A, b, mode): ## write a separate function for D-SDP feasibility test?
	# reshape feasible set
	m = len(b)
	n = C.shape[0] # square
	ys = [np.random.choice((-1,1))*np.random.random() for i in range(m)] # len(ys) == m
	#print("ys = ", ys)
	Ss = C - sum([ys[i]*A[i] for i in range(m)], np.zeros((n,n)))

	# choose y*_{m+1} to make S* strictly positive definite
	smallest_eig = min(np.linalg.eigvals(Ss))
	#print('smallest_eig = ', smallest_eig)
	if smallest_eig <= 0:
	    ymp1 = smallest_eig - 1
	    Amp1 = np.eye(n)
	else:
	    ymp1 = 0
	    Amp1 = np.zeros((n,n))
	if mode == 2: # feasibility_dual mode
		if ymp1 == 0 and ys[m-1] > 0: # ys[m-1] is ys_{m+1}
			print('ys_(m+1) = {0} > 0 and hence SDP-D is strictly feasible'.format(ys[m-1]))
			print('Early exit in ' + find_start_points_optimum.__name__)
			exit() ## using exit()? since the task is DONE!

	# compute S*, X*, b*
	#print("ymp1 = ", ymp1)
	#print("Ss = ", Ss)
	Ss -= ymp1*Amp1 # make it to be "-"
	#print("Ss =", Ss)
	Xs = np.linalg.inv(Ss)
	# A[i]*Xs: elem-wise mult. for np.ndarray
	bs = [(A[i]*Xs).sum() for i in range(m)]
	#print("Xs =", Xs)
	#print("bs = ", bs)

	return Xs,ys,bs,ymp1

def compute_optimum(C, A, b, mode):
	# compute start points
	# C, A are np.2darray (full dense matrix), b is a list of float
	# Xs: np.ndarray, ys, bs: list, ymp1 = scaler (nonpositive!)
	# print_data(C,A,b)

	Xs,ys,bs,ymp1 = find_start_points_optimum(C, A, b, mode)
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

def compute_feasibility_dual(C, A, b, mode):
	# max y_{m+1} s.t. C - \sum Ai*yi - I*y_{m+1} >= 0
	# => b~ = [0;...;0;1] \in \R^{m+1}
	## enlarge b and A, A[m+1] = I

	n = C.shape[0] # square

	# first check linear dependence of A_i
	# this step can be skipped if A_i are already linearly indepenent
	# but there is no harm doing it for small test problem
	linear_depend_info1 = test_linear_depend(A)
	if linear_depend_info1:
		print('Please remove redundacy between A_i (i=1,...,m) before running this code!')
		print('Early exit in ' + compute_feasibility_dual.__name__)
		exit()

	# second check whether I is a linear combination of A_i's
	# if so, we come to a conclusion: SDP-D is strictly feasible (can be proved straightforwardly!)
	# no need to continue computing in this case!
	linear_depend_info2 = test_linear_depend(A, np.eye(n))
	if linear_depend_info2:
		print('Identity matrix is a linear combination of A_i (i=1,...,m)')
		print('Hence SDP-D is strictly feasible!')
		print('Early exit in ' + compute_feasibility_dual.__name__)
		exit()
	else:
		print('We can safely append A with I and modify b, continue...')

	# modify b, add one more dimension, but this change only affect inside comp_feasibiility
	b = [0]*len(b) + [1] # make it to be "+"
	# modify A, add one more dimension
	A.append(np.eye(n)) # this modification will affect outside this function if A (not A[:]) is passed in
	# print_data(C,A,b)
	X,y,S = compute_optimum(C,A,b,mode)

	return X,y,S # len(y) == m+1

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
	print('')


# main function (for test purpose)
if __name__ == '__main__':
	## set up parameters here
	## cf. /examples/readme.txt for more details
    ## mode 1: optimum_solve
    ## mode 2: feasibility_dual
    ## mode 3: feasibility_primal
    ## mode 4: other test mode
	example_tag = '6' ## change here, also can be input on command line
	mode = 2
    
    # read data from file located in examples: C, A, b
	cwd = os.getcwd()
	example_dirname = os.path.join(cwd, 'examples')
	C,A,b = sdp_in.read_in_SDP(example_dirname, example_tag)
    ## do the test to modify A, b if necessary: remove redunacy of A_i? -- not the goal of this project

    # task 1: optimum solve
	if mode == 1:
		X,y,S = compute_optimum(C, A, b, mode)
	    # postprocessing
	    # print_data(C,A,b)
		print('\n---------------Postprocessing: for small problem-------------------')
		postprocess(X,y,S,C,A[:],b) # pass a copy of A: A[:]
		# C,A,b will not change after this call
		# if the data A is large, just pass A as "reference"
		## fix this later to make it more clear and efficient!

    # task 2: feasibility_dual
	if mode == 2:
		print('\n---------------Feasibility Test for SDP_D-------------------')
		X,y,S = compute_feasibility_dual(C, A[:], b, mode) # len(b) == m
		## output: len(y) == m+1
		last_idx = len(b) # == len(y) - 1
		eps = 1.0e-10 # change here if necessary
		print("max y_m+1 = {0}, eps = {1}".format(y[last_idx], eps))
		if ( y[last_idx] > eps):
			print("Dual of SDP is strict feasible!")
		elif (y[last_idx] < -eps):
			print("Dual of SDP is infeasible!")
		else: #==0?
			print("Two cases for Dual of SDP:\n feasible, but not strict feasible \n OR infeasible")

	# task 4: other test
	if mode == 4:
		A = np.array([[1,2,3], [4,5,6], [7,8,9]])
		print(type(A))
		print(A)
		a = A[np.triu_indices(3)]
		print(type(a))
		print(a)
		M = np.zeros((2,6))
		M[0,:] = a
		print(M)
		#A = numpy.vstack([A, newrow])
		M = np.vstack([M, a])
		print(M)
