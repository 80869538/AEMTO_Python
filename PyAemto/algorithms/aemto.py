import numpy as np
import sys
from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from copy import copy, deepcopy


sys.path.append('../../')
from PyAemto.problems.mtobenchmark.rastrigin_T import Rastrigin_T
from PyAemto.problems.mtobenchmark.griewank_T import Griewank_T

from PyAemto.core.Tasks import Tasks as Tasks
import PyAemto.core.ProblemInfo as ProbInfo

EPS = 1e-10
run_id = 0
p_tsf_lb = 0.05  # lower bound of transfer probability
p_tsf_ub = 0.70  #upper ...
alpha=0.3 #reward update rate
pbase=0.3
ntasks = 2 #test purpose
Gmax = 1 #max generation
MTO = True #enable knowledge transfer between tasks
pop_size = 100

tasks_tsf_cnt = np.array([],dtype=np.uint32)

task_evals = [] #evaluator of each task
problem_infos = []

tasks = [] #population of each task
record_tasks = [] #record of each task

ret = np.array([])

def AEMTO(problems, algorithms, p_tsf_ub, p_tsf_lb, MTO, alpha):
    tasks = Tasks(problems, algorithms, p_tsf_ub, p_tsf_lb, MTO, alpha)
    
    for g in range(Gmax):
        #This step used task selection prob directly as task fintness 
        #to select a group of tasks to select from, tasks with higher 
        #selection prob might be selected multiple times. 

        #upon checking this code used in the implementation, I found it's the 
        #result is the same as just use pop_size * pdf

        #transfer with prob pdf 
        need_transfer = np.random.random_sample(len(problems)) <= tasks.Tpdf
        
        T = np.full_like(tasks.Spdf,0.0) #transfer matrix (n_task, n_task)
        T[need_transfer, ...] = tasks.Spdf[need_transfer, ...] 
        assert(np.diag(T).sum() == 0) #diag should keep zero, no self transfer

        P_dist = np.sum(T,axis=1,keepdims=True)/pop_size #calculate pointer distance
        start = np.random.uniform(0,P_dist) #first pointer
        pointers = np.array([start + i*P_dist for i in range(0, pop_size)]) #calculate pointers, (n_pointers, n_task, n_task)
        fitness_sum = np.add.accumulate(T,1) #sum of probability/fitness for tasks, construct the wheel, (n_task,n_task)

        #find those pointers that fall inside a wheel area
        t = fitness_sum > pointers #find when > is true (n_pointers, n_task, n_task)
        S = np.diff(t.sum(axis = 0),prepend=0) #count true value, since result is accumulated, find diff
        assert(np.diag(S).sum() == 0) #diag should keep zero, no self transfer
        print(S)

        selected = Population()

        tasks.eval_task(0,S)
        tasks.reuse(0)







        





problem1 = Griewank_T(n_var=50)
problem2 = Rastrigin_T(n_var=50)
problem3 = Rastrigin_T(n_var=50)
algorithm =  DE(
        pop_size=pop_size,
        sampling=LHS(),
        variant="DE/rand/1/bin",
        CR=0.9,
        dither="vector",
        jitter=False
    )
problems = [problem1, problem2,problem3]
algorithms = list(map(lambda problem:deepcopy(algorithm).setup(problem),problems))

AEMTO(problems, algorithms, p_tsf_ub, p_tsf_lb, MTO, alpha = alpha)



            

