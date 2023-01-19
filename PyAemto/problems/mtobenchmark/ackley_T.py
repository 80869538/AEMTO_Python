import numpy as np

import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem



class Ackley(Problem):

    def __init__(self, n_var=2, a=20, b=1/5, c=2 * np.pi):
        super().__init__(n_var=n_var, n_obj=1, xl=-50.0, xu=+50.0, vtype=float)
        self.a = a
        self.b = b
        self.c = c

    def _evaluate(self, x, out, *args, **kwargs):
        part1 = -1. * self.a * anp.exp(-1. * self.b * anp.sqrt((1. / self.n_var) * anp.sum(x * x, axis=1)))
        part2 = -1. * anp.exp((1. / self.n_var) * anp.sum(anp.cos(self.c * x), axis=1))
        out["F"] = part1 + part2 + self.a + anp.exp(1)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)

class Ackley_T(Ackley):
    def __init__(self, M=None, o=None, PF = 0, PS = None, n_var=2, a=20, b=1/5, c=2 * np.pi):
        #M: np.array matrix
        #o: np.array vector
        super().__init__(n_var=n_var, a=a, b=b, c=c)


        self.M = M if M else np.identity(self.n_var)
        self.o = o if o else np.full(self.n_var, 0)
        self.PF = PF
        self.PS = PS if PS else np.full(self.n_var, 0)

    def _transform(self, x):
        #x: np.array vector
        return np.transpose(np.dot(self.M, np.transpose(x - self.o)))

    def _evaluate(self, x, out, *args, **kwargs):
         x = self._transform(x)
         super()._evaluate(x, out, *args, **kwargs)
    
    def _calc_pareto_front(self):
        return self.PF

    def _calc_pareto_set(self):
        return self.PS

        
    
    

    
