from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.util.misc import random_permuations
import numpy as np
from pymoo.core.individual import Individual
from copy import copy, deepcopy


class DE(GeneticAlgorithm):
    def __init__(self,
                pop_size=100,
                n_offsprings=None,
                sampling=FloatRandomSampling(),
                output=SingleObjectiveOutput(),
                **kwargs
                ):
        super().__init__(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=sampling,
            survival=None,
            output=output,
            eliminate_duplicates=False,
            **kwargs)
    
    def _initialize_advance(self, infills=None, **kwargs):
        FitnessSurvival().do(self.problem, self.pop, return_indices=True)
    
    def CheckBound(self,to_check_elements,min_bound,max_bound):
        while to_check_elements < min_bound or to_check_elements > max_bound:
            if to_check_elements < min_bound:
                to_check_elements = min_bound + (min_bound - to_check_elements)
            if to_check_elements > max_bound:
                to_check_elements = max_bound - (to_check_elements - max_bound)
    
    def _infill(self):
        F = 0.5
        CR = 0.9
        update_num = 0
        for i in self.pop:
            tmp_individual = deepcopy(i)

            r = random_permuations(1,self.pop_size)
            k = np.random.randint(0, 49)
            for j in range(self.problem.n_var):
                if j == k or np.random.uniform() < CR:
                    tmp_individual.X[j] = self.pop[r[0]].X[j] + F * (self.pop[r[1]].X[j] - self.pop[r[2]].X[j])
                tmp_individual.X[j] = self.CheckBound(tmp_individual.X[j], 0.0, 1.0)

            tmp_individual_F = self.problem.evaluate(tmp_individual.X)
            tmp_individual.set("F",tmp_individual_F)

        out = self.problem.evaluate(tmp_individual.get("X"))
        tmp_individual.set("F", out)
        tmp_individual.set("F",tmp_individual_F)
        if tmp_individual.F < i.F:
            i = tmp_individual
            update_num += 1
            print("update_num")
            print(update_num)

        return self.pop

    def _advance(self, infills=None, **kwargs):
        pass






