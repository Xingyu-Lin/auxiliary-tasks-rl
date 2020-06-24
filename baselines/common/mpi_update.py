from mpi4py import MPI
import baselines.common.tf_util as U
import numpy as np


class MpiUpdate(object):
    def __init__(self, var_list, *, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.scale_grad_by_procs = scale_grad_by_procs
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None else comm

    def update(self, localg):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        globalg = np.zeros_like(localg)
        self.comm.Allreduce(localg, globalg, op=MPI.SUM)
        if self.scale_grad_by_procs:
            globalg /= self.comm.Get_size()

        self.t += 1
        self.setfromflat(np.clip(self.getflat() + globalg, 0., None))

    def set(self, local_param):
        if self.t % 100 == 0:
            self.check_synced()
        local_param = local_param.astype('float32')
        global_param = np.zeros_like(local_param)
        self.comm.Allreduce(local_param, global_param, op=MPI.SUM)
        if self.scale_grad_by_procs:
            global_param /= self.comm.Get_size()

        self.t += 1
        self.setfromflat(np.clip(global_param, 0., None))

    def sync(self):
        weight = self.getflat()
        self.comm.Bcast(weight, root=0)
        self.setfromflat(weight)

    def check_synced(self):
        if self.comm.Get_rank() == 0:  # this is root
            weight = self.getflat()
            self.comm.Bcast(weight, root=0)
        else:
            weightlocal = self.getflat()
            weightroot = np.empty_like(weightlocal)
            self.comm.Bcast(weightroot, root=0)
            assert (weightroot == weightlocal).all(), (weightroot, weightlocal)
