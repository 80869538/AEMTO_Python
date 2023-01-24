import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem


class Rastrigin(Problem):
    def __init__(self, n_var=2, A=10.0):
        super().__init__(n_var=n_var, n_obj=1, xl=0.0, xu=1.0, vtype=float)
        self.A = A

    def _evaluate(self, x, out, *args, **kwargs):
        z = anp.power(x, 2) - self.A * anp.cos(2 * anp.pi * x)
        out["F"] = self.A * self.n_var + anp.sum(z, axis=1)

    def _calc_pareto_front(self):
        return 0.0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)

class Rastrigin_T(Rastrigin): #TO_DO better to do it by inherient from two classes, one transform, the other otginal problem
    def __init__(self, M=None, o=None,PF = 0, l=-50, u=50 ,PS = None,n_var=50, A=10.0):
        super().__init__(n_var=n_var)
        self.A = A
        self.M = M if M is not None else np.identity(self.n_var)
        self.o = o if o is not None else np.full(self.n_var, 0)
        self.PF = PF
        self.PS = PS if PS else np.full(self.n_var, 0)
        self.l =l
        self.u = u
        self.scale_rate_ = (self.u - self.l) / (self.xu - self.xl)
        self.bias_vec_ = np.full(50, -self.scale_rate_ * self.xl + self.l)

    
    
    def _transform(self, x):
        #x: np.array vector
        return np.transpose(np.dot(self.M, np.transpose( self.scale_rate_ * (x - self.xl) + self.bias_vec_ - self.o)))

    def _evaluate(self, x, out, *args, **kwargs):
        x = self._transform(x)
        super()._evaluate(x, out, *args, **kwargs)


    def _calc_pareto_front(self):
        return self.PF

    def _calc_pareto_set(self):
        return self.PS
