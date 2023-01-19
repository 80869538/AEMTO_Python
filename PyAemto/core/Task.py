from pymoo.optimize import minimize
import sys
from copy import copy, deepcopy
from pymoo.core.population import Population

sys.path.append('../../')

from PyAemto.operators.fps import FPS
class Task:
    def  __init__(self, problem, algorithm, spdf={} ):
        self.problem = problem
        self.algorithm = algorithm
    
    def initalize_infill(self):
        if self.algorithm.pop == None:
            self.algorithm.next()
        else:
            return
    
    #when other task select, need deepcopy so their population are not affected
    def select(self, num_sel):
        if num_sel > 0:
            pop = deepcopy(self.algorithm.pop)
            return FPS(pop, num_sel)
        else:
            return Population.empty()
    
    def eval(self, pop):
        self.algorithm.evaluator.eval(self.problem, pop, algorithm=self.algorithm)

    
    def get_pop(self):
        pop = self.algorithm.pop
        return pop
        
    def solve(self):
        res = minimize(self.problem, self.algorithm,
               ('n_gen', 10),
               seed=1,
               verbose=True)
        return res