import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem


class Rastrigin(Problem):
    def __init__(self, n_var=2, A=10.0):
        super().__init__(n_var=n_var, n_obj=1, xl=-50, xu=50, vtype=float)
        self.A = A

    def _evaluate(self, x, out, *args, **kwargs):
        z = anp.power(x, 2) - self.A * anp.cos(2 * anp.pi * x)
        out["F"] = self.A * self.n_var + anp.sum(z, axis=1)

    def _calc_pareto_front(self):
        return 0.0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)

class Rastrigin_T(Rastrigin): #TO_DO better to do it by inherient from two classes, one transform, the other otginal problem
    def __init__(self, M=None, o=None,PF = 0, PS = None,n_var=50, A=10.0):
        super().__init__(n_var=n_var)
        self.A = A
        self.M = M if M else np.identity(self.n_var)
        self.o = o if o else np.full(self.n_var, 0)
        self.PF = PF
        self.PS = PS if PS else np.full(self.n_var, 0)
    
    def _transform(self, x):
        #x: np.array vector
        return np.transpose(np.dot(self.M, np.transpose(x - self.o)))

    def _evaluate(self, x, out, *args, **kwargs):
        z = anp.power(x, 2) - self.A * anp.cos(2 * anp.pi * x)
        out["F"] = self.A * self.n_var + anp.sum(z, axis=1)

    def _calc_pareto_front(self):
        return self.PF

    def _calc_pareto_set(self):
        return self.PS
