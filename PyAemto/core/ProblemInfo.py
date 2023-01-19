class ProblemInfo:
    def __init__(self, task_id, dim, calc_dim,benchfunc_name,shift_data_file,rotation_data_file,arm_data_file,max_bound=1.0,min_bound=0.0):
        super().__init__()
        self.task_id = task_id
        self.dim = dim
        self.calc_dim = calc_dim
        self.benchfunc_name = benchfunc_name

        self.shift_data_file = shift_data_file
        self.rotation_data_file = rotation_data_file
        self.arm_data_file = arm_data_file

