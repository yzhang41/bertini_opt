Example list:
  
  Ex1: Ex1 in http://www.math.nus.edu.sg/~mattohkc/sdpt3/sdpexample.html
    data format: A1.txt, C1.txt, b1.txt
    For this example, SDP-P is only feasible (only one feasible pt), and SDP-D is strictly feasible, so SDP-P optimum is attained from optimization theory.

  Ex2: Ex2 in http://www.math.nus.edu.sg/~mattohkc/sdpt3/sdpexample.html

  Ex3: Example 2.11 in p.11 of the book “Semidefinite Optimization and Convex Algebraic Geometry”

  Ex4: Example 2.14 with alpha=1 in p.14 of the book “Semidefinite Optimization and Convex Algebraic Geometry”

  Ex5: from Jon with alpha = 0.1, only for feasibility test (dual), so bfile is unnecessary

  Ex6: from Jon with alpha = csrt(2), only for feasibility test (dual), so bfile is unnecessary

  Ex7: n = 2, m = 1, SDP-P optimal is NOT attained but SDP-D optimal is achieved. In fact SDP-P is strictly feasible (feasibility_primal_test fails so far, need to switch to user_homotopy: 2) We can also show that SDP-D is not strictly feasible (only feasible) (cf. feasibility_dual_test).
    Remark: optimum_run also fails, b/c of inf. solu.
            If we switch to homotopy:2, the result is still NOT good enough! Why?
