Example list:
  
  Ex1: Ex1 in http://www.math.nus.edu.sg/~mattohkc/sdpt3/sdpexample.html
    data format: A1.txt, C1.txt, b1.txt
    For this example, SDP-P is only feasible (only one feasible pt), and SDP-D is strictly feasible, so SDP-P optimum is attained from optimization theory.

  Ex2: Ex2 in http://www.math.nus.edu.sg/~mattohkc/sdpt3/sdpexample.html

  Ex3: Example 2.11 in p.11 of the book “Semidefinite Optimization and Convex Algebraic Geometry”

  Ex4: Example 2.14 with alpha=1 in p.14 of the book “Semidefinite Optimization and Convex Algebraic Geometry”, duality gap = alpha.

  Ex5: from Jon with alpha = 0.1, only for feasibility test (dual), so bfile is unnecessary

  Ex6: from Jon with alpha = csrt(2), only for feasibility test (dual), so bfile is unnecessary

  Ex7: n = 2, m = 1, SDP-P optimal is NOT attained but SDP-D optimal is achieved. In fact SDP-P is strictly feasible (feasibility_primal_test fails so far, need to switch to user_homotopy: 2) We can also show that SDP-D is not strictly feasible (only feasible) (cf. feasibility_dual_test).
    Remark: switch to homotopy:2, mode 1 outputs complex solution
                                  mode 2 works ok
                                  mode 3 fails, why?

  Ex8: n = 2, m = 2, use this example to test feasibility_test_primal, since the optimal lambda is 0 but not achievable! see what will happen. 
    Remark: For this example, no matter how to choose C, X = [0,1; 1,x_22], hence det(X) = -1<0. So SDP-P is infeasible!
            switch to homotopy:2, mode 1 outputs complex solution
                                  mode 3 outputs complex solution
  Remark of Ex7 and Ex8:
    — we need to use different variable_group for X, y and possible lambda
    - for Ex7, we know the “optimal” X_22 is +\infty, if we set X_22 as a separate 
      variable_group, we can recover a very good “solution” for SDP-P.
      If we set variable_group X_11, X_12, X_22, then X_11 and X_12 might not be accurate
      because of dehomogenization.

