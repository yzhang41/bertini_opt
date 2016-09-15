"""
Remark: 1. path tracking of solus to homogopy, all real solu?
        2. three modes:
        	mode 1: optimum_solve
        	mode 2: feasibility_dual
        	mode 3: feasibility_primal
        	mode 4: test
        3. add feasibility_primal mode -- Done!
        4. "SDP-P strict feasible" IS NOT EQUIVALENT TO "SDP-D strict feasible", cf. Example 1
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

	# enlarge ys for mode 3 (feasibility_primal)
	if mode == 3:
		#temp = np.random.random()
		#print(-abs(temp))
		#ys.append(-abs(temp)) # make it negative!!! ??? why???
		ys.append(np.random.choice((-1,1))*np.random.random())
		#print(temp)
		#ys.append(0.78)
	# now len(ys) = m+1, the last variable represents lambda

	# choose y*_{m+1} to make S* strictly positive definite
	smallest_eig = min(np.linalg.eigvals(Ss))
	#print('smallest_eig = ', smallest_eig)
	if smallest_eig <= 0:
	    ysmp1 = smallest_eig - 1
	    Amp1 = np.eye(n)
	else:
	    ysmp1 = 0
	    Amp1 = np.zeros((n,n))
	if mode == 2: # feasibility_dual mode
		if ysmp1 == 0 and ys[m-1] > 0: # ys[m-1] is origional ys_{m+1}
			print('ys_(m+1) = {0} > 0 and hence SDP-D is strictly feasible'.format(ys[m-1]))
			print('Early exit in ' + find_start_points_optimum.__name__)
			exit() ## using exit()? since the task is DONE!

	# compute S*, X*, b*
	#print("ysmp1 = ", ysmp1)
	#print("Ss = ", Ss)
	Ss -= ysmp1*Amp1 # make it to be "-"
	# print(Ss)
	# print('smallest_eig = ', min(np.linalg.eigvals(Ss)))
	# print('trace(Ss) = ', np.trace(Ss))
	# if mode == 3: # scale Ss s.t. trace(Ss) == 2
	# 	#Ss = Ss/(np.trace(Ss)/2) # check here
	# 	Ss /= np.trace(Ss)/2
	# 	print('smallest_eig = ', min(np.linalg.eigvals(Ss)))
	# 	### NEED THIS! IMPORTANT!
	# 	for i in range(m): # range(m) instead of range(m+1)!!!
	# 		ys[i] /= np.trace(Ss)/2
	# 		ysmp1 /= np.trace(Ss)/2
	# 	print(ys[:m])
	# 	#ys /= np.trace(Ss)/2 # scale ys also, otherwise equality for Ss fails!!!
	# 	if abs(np.trace(Ss) - 2.0) > 1.0e-8:
	# 		print('trace(Ss) = {0}, Error of trace(Ss) in mode {1}, early exit!'.format(np.trace(Ss), mode))
	# 		exit()

	#print("Ss =", Ss)
	#print(Ss)
	Xs = np.linalg.inv(Ss)
	# if mode == 3: # mode 3
	# 	Xs -= ys[m]*np.eye(n)
	# A[i]*Xs: elem-wise mult. for np.ndarray
	bs = [(A[i]*Xs).sum() for i in range(m)]
	if mode == 3:
		bs = [bs[i] - ys[m]*np.trace(A[i]) for i in range(m)]
	#print("Xs =", Xs)
	#print("bs = ", bs)
	trace_Ss = np.trace(Ss)

	if mode == 3:
		return Xs,ys,bs,ysmp1,trace_Ss # for mode 3, len(ys) = m+1 instead of m
	else:
		return Xs,ys,bs,ysmp1

def compute_optimum(C, A, b, mode):
	# compute start points
	# C, A are np.2darray (full dense matrix), b is a list of float
	# Xs: np.ndarray, ys, bs: list, ysmp1 = scaler (nonpositive!)
	# print_data(C,A,b)

	if mode == 3:
		Xs,ys,bs,ysmp1,trace_Ss = find_start_points_optimum(C, A, b, mode)
	else:
		Xs,ys,bs,ysmp1 = find_start_points_optimum(C, A, b, mode)
	m = len(b) # use this, b/c in mode 3, len(y) is added \lambda at last position
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
	constants["ysmp1"] = ysmp1 # use this bc y_m is reserved for \lambda in mode 3

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

	# modify first set of eqns. for mode 3
	if mode == 3:
		for i in range(m):
			eqterms = []
			for j in range(n):
				eqterms.append("y{0} * A{1}_{2}_{2}".format(m,i,j))
			add_terms = " - " + " - ".join(eqterms)
			functions["f{0}".format(i)] += add_terms

	# for mode 3, add one more eqn to f[m]: S \cdot I - 1 - \mu * trace(Ss) = 0
	# add variable lambda to yvariables y[m]
	if mode == 3:
		yvariables.append("y{0}".format(m))
		eqterms = [] # waster a lot of old eqterms??? modify later for efficiency
		for i in range(n):
			eqterms.append("S{0}_{0}".format(i))
		#print(eqterms)
		functions["f{0}".format(m)] = "+".join(eqterms) + " - 1 - mu * {0}".format(trace_Ss-1.0)

	# second set ((n+1)*n/2 eqns): S.dot(X) - \mu * Id
	# for mode 3, it is S.dot(X + \lambda I) - \mu * Id
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
	        # if mode == 3:
	        # 	h += " + y{0}*S{1}_{2}".format(m,i,j) # y[m] is \lambda
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
	
	#ycoords = optimum.coordinates[-m:]
	ycoords = optimum.coordinates[int(n*(n+1)/2):]
	# if mode == 3:
	# 	lambd = optimum.coordinates
	
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
	
	# if mode == 3:
	# 	return X,lambd,y,S # return np.matrix (X,S), scaler lambd, np.array y
	# else:
	return X,y,S # return np.matrix, np.array
	# in mode 3, y[m] is lambda!

def compute_feasibility_dual(C, A, b, mode):
	# max y_{m+1} s.t. C - \sum Ai*yi - I*y_{m+1} >= 0
	# => b~ = [0;...;0;1] \in \R^{m+1}
	## enlarge b and A, A[m+1] = I

	n = C.shape[0] # square
	# check whether I is a linear combination of A_i's
	# if so, we come to a conclusion: SDP-D is strictly feasible (can be proved straightforwardly!)
	# no need to continue computing in this case!
	linear_depend_info = test_linear_depend(A, np.eye(n))
	if linear_depend_info:
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

def compute_feasibility_primal(C, A, b, mode):
	"""
	min \lambda { X + \lambda I >= 0, A_i \cdot X = b_i, i=1,...,m }
	add one more primal variable \lambda, this new variable will append to the last of yvariables
	this will bring one more constraint eqn. into the KKT system: S \cdot I - 1 = 0
	Set C = 0
	Remark: note that here we use X to represent X + \lambda I, hence we need to modify the first set of eqns.
	"""
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
	## mode 1: optimum_solve
	## mode 2: feasibility_dual
	## mode 3: feasibility_primal
	## mode 4: other test mode
	mode_dict = { '1':'optimum_solve', '2':'feasibility_test_dual', '3':'feasibility_test_primal', '4':'other test' }
	example_tag = '1' ## change here, also can be input on command line
	mode = 3

	print('---------------Example {0}-------------------'.format(example_tag))
	print('---------------mode {0}: {1}-------------------'.format(mode, mode_dict[str(mode)]))

    # read data from file located in examples: C, A, b
	cwd = os.getcwd()
	example_dirname = os.path.join(cwd, 'examples')
	C,A,b = sdp_in.read_in_SDP(example_dirname, example_tag)
    ## do the test to modify A, b if necessary: remove redunacy of A_i? -- not the goal of this project

	# task 0: first check linear dependence of A_i
	# this step can be skipped if A_i are already linearly indepenent
	# but there is no harm doing it for small test problem
	linear_depend_info = test_linear_depend(A)
	if linear_depend_info:
		print('Please remove redundacy between A_i (i=1,...,m) before running this code!')
		print('Early exit in ' + __name__)
		exit()

    # task 1: optimum solve
	if mode == 1:
		X,y,S = compute_optimum(C, A, b, mode)
		# postprocessing
		# print_data(C,A,b)
		print('\n---------------Output: postprocessing: for small problem-------------------')
		postprocess(X,y,S,C,A[:],b) # pass a copy of A: A[:]
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
		optimal_primal_value = np.multiply(np.matrix(C),X).sum()
		print("optimal primal obj. value: C \cdot X = {0}". format(optimal_primal_value))
		print("duality gap = {0}".format(optimal_primal_value - y[last_idx]))
		print("eps = {0}\n".format(eps))
		if ( y[last_idx] > eps):
			print("Dual of SDP is strict feasible!")
		elif (y[last_idx] < -eps):
			print("Dual of SDP is infeasible!")
		else: #==0?
			print("Two possible cases for Dual of SDP:\n feasible, but not strict feasible \n OR infeasible")

	if mode == 3: # feasibility_primal
		X,y,S = compute_feasibility_primal(C, A[:], b, mode)
		print('\n---------------Output: feasibility Test for SDP_P-------------------')
		n = C.shape[0]
		m = len(b) # len(y) == m+1
		#print(y)
		lambd = y[m]
		optimal_dual_value = np.dot(b, y[:m]) # check here
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
		#print_data(X-y[m]*np.eye(n), A, b)
		#print(X)
		#print(S)
		### THIS IS the real X we want in SDP-P!
		print(X - y[m]*np.eye(n))
		print(min(np.linalg.eigvals(X - y[m]*np.eye(n))))
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

	print('')
