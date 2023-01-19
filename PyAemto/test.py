import numpy as np

# from problems.mtobenchmark.rastrigin_T import Rastrigin_T
# from pymoo.visualization.fitness_landscape import FitnessLandscape

# problem = Rastrigin_T(n_var=2)
# FitnessLandscape(problem, angle=(45, 45), _type="surface").show()

A = np.array([1,2,3,5,6,7])
B = np.array([[0,1],[2,3]])
print(A[B])
