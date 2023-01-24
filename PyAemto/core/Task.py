from pymoo.optimize import minimize
import sys
from copy import copy, deepcopy
from pymoo.core.population import Population
from pymoo.util.roulette import RouletteWheelSelection

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
            self.algorithm.evaluator.eval(self.problem, pop, algorithm=self.algorithm)
            select = RouletteWheelSelection(pop.get("F"),False)
            # return FPS(pop, num_sel)
            return pop[select.next(num_sel)]
        else:
            return Population.empty()
    
    def eval(self, pop):
        self.algorithm.evaluator.eval(self.problem, pop, algorithm=self.algorithm)

    
    def get_pop(self):
        pop = self.algorithm.pop
        return pop
    
    def result(self):
        return self.algorithm.result()
    
    def next(self):
        self.algorithm.next()
