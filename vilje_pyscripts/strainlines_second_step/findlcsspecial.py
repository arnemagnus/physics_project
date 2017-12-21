# The numerical integrators are located in a module two levels above
# the current working directory. Hence:
import sys
sys.path.insert(0, '..')


# Numpy
import numpy as np

# Numba (JiT)
from numba import njit

# (Primitive) timing functionality
import time

# Multiprocessing:
import multiprocessing as mp

# Spline interpolation:
from scipy.interpolate import RectBivariateSpline

# Check whether folders exist or not, necessary
# for storing advected states:
import os
import errno

# Function that makes a directory if it does not exist,
# and raises an exception otherwise
# (necessary for storing advected states)

def ensure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

fixed_step_integrators = set(['euler','rk2','rk3','rk4'])

from numerical_integrators.adaptive_step import rkbs32,rkbs54,rkdp54,rkdp87
from numerical_integrators.singlestep import euler,rk2,rk3,rk4
t_start = 0.
t_end = 20.
h = 0.1

atol = 1e-1
rtol = 1e-1

integrator=rkbs54

x_min,x_max = 0,2
y_min,y_max = 0,1

nx,ny = 1000,500

class Strainline:
    def __init__(self,startpoint,l_min,l_f_max,lmbd2_spline):
        self.pos = np.array([startpoint]).reshape((2,1))
        self.l_min = l_min
        self.lmbd2_spline = lmbd2_spline
        self.stationary = np.zeros(2,dtype=np.bool)
        self.cont_failure = np.zeros(2,dtype=np.bool)
        self.outs_dom = np.zeros(2,dtype=np.bool)
        self.max_iter = np.zeros(2,dtype=np.bool)
        self.startpoint_index = 0
        self.tailcut_start = 0
        self.tailcut_end = 0
    def long_enough(self):
        return self.len >= self.l_min
    def append(self,pos):
        self.pos = np.hstack((self.pos,pos.reshape((2,1))))
    def traj(self):
        return self.pos
    def avg_lmbd2(self):
        return np.mean(self.lmbd2_spline.ev(self.pos[1],self.pos[0]))
    def lngth(self):
        return np.sum(np.sqrt((np.diff(self.pos,axis=1)**2).sum(axis=0)))
    def tailcut_traj(self):
        return self.pos[:,self.tailcut_start:self.tailcut_end]
    def tailcut_avg_lmbd2(self):
        return np.mean(self.lmbd2_spline.ev(self.tailcut_traj()[1],self.tailcut_traj()[0]))
    def tailcut_lngth(self):
        return np.sum(np.sqrt((np.diff(self.tailcut_traj(),axis=1)**2).sum(axis=0)))

max_iter = 10000
stride = 0.001
l_f_max = 0.2
l_min = 1.
tol_alpha = 1.e-6

try:
    if integrator.__name__ in fixed_step_integrators:
        strainlines = np.load('precomputed_strainlines/{}/strainlines_h={}_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.npy'.format(integrator.__name__,h,max_iter,stride,l_f_max,l_min,tol_alpha))
    else:
        strainlines = np.load('precomputed_strainlines/{}/strainlines_atol={}_rtol={}_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.npy'.format(integrator.__name__,atol,rtol,max_iter,stride,l_f_max,l_min,tol_alpha))
except IOError:
    print('Strainline config not loaded!')

def find_intersections(strainlines,x_min,x_max,y_min,y_max,num_horz,num_vert,vert_x,horz_y):
    n_strainlines = len(strainlines)
    isect_horz = [[[[],[]] for i in range(num_horz)] for j in range(n_strainlines)]
    isect_vert = [[[[],[]] for i in range(num_vert)] for j in range(n_strainlines)]
    #tic = time.time()
    for i in range(n_strainlines):
        #traj = strainlines[i].traj()
        traj = strainlines[i].tailcut_traj()
        for j in range(num_horz):
            for k in range(np.size(traj,1)-1):
                if (traj[1,k]-horz_y[j])*(traj[1,k+1]-horz_y[j])<=0:
                    wk = (traj[1,k+1]-horz_y[j])/(traj[1,k+1]-traj[1,k])
                    isect_horz[i][j][0].append(wk*traj[0,k]+(1-wk)*traj[0,k+1])
                    isect_horz[i][j][1].append(horz_y[j])
        for j in range(num_vert):
            for k in range(np.size(traj,1)-1):
                if (traj[0,k]-vert_x[j])*(traj[0,k+1]-vert_x[j])<=0:
                    wk = (traj[0,k+1]-vert_x[j])/(traj[0,k+1]-traj[0,k])
                    isect_vert[i][j][0].append(vert_x[j])
                    isect_vert[i][j][1].append(wk*traj[1,k]+(1-wk)*traj[1,k+1])
    #mins,secs = np.divmod(time.time()-tic,60)
    #print('Intersections found in {} minutes and {} seconds'.format(mins,secs))
    return isect_horz,isect_vert

def find_neighbors(n_strainlines,num_horz,num_vert,intersections_horz,intersections_vert,l_n,n_proc=4):
    div = np.floor(np.linspace(0,n_strainlines,n_proc+1)).astype(int)
    queues = [mp.Queue() for j in range(n_proc)]
    processes = [mp.Process(target=_find_neighbors,
                           args=(n_strainlines,num_horz,num_vert,intersections_horz,intersections_vert,
                                l_n,div[j],div[j+1],j,queues[j]))
                for j in range(n_proc)]
    nbrs_vert = []
    nbrs_horz = []
    #tic = time.time()
    for process in processes:
        process.start()
    for q in queues:
        nbrs_vert += q.get()
        nbrs_horz += q.get()
    for process in processes:
        process.join()
    #mins,secs = np.divmod(time.time()-tic,60)
    #print('Neighbors found in {} minutes and {} seconds'.format(mins,secs))
    return nbrs_vert,nbrs_horz

def _find_neighbors(n_strainlines,num_horz,num_vert,intersections_horz,intersections_vert,l_n,n0,nend,pn,q):
    #n_strainlines = len(strainlines)
    nbrs_vert = []
    nbrs_horz = []
    for i in range(n0,nend):
        nbrs_strline_vert = []
        for j in range(num_vert):
            nbrs_vert_isect = [[] for k in range(len(intersections_vert[i][j][0]))]
            for k in range(n_strainlines):
                for m in range(len(intersections_vert[i][j][0])):
                    for n in range(len(intersections_vert[k][j][0])):
                        if k!=i and np.abs(intersections_vert[i][j][1][m]-intersections_vert[k][j][1][n]) < l_n:
                            nbrs_vert_isect[m].append(k)
                #for m in range(np.minimum(np.size(intersections_vert[i][j][0]),np.size(intersections_vert[k][j][0]))):
                #    if k != i and np.abs(intersections_vert[i][j][1][m]-intersections_vert[k][j][1][m]) < l_n:
                #        nbrs_vert_isect[m].append(k)
            nbrs_strline_vert.append(nbrs_vert_isect)
        nbrs_vert.append(nbrs_strline_vert)

        nbrs_strline_horz = []
        for j in range(num_horz):
            nbrs_horz_isect = [[] for k in range(len(intersections_horz[i][j][0]))]
            for k in range(n_strainlines):
                for m in range(len(intersections_horz[i][j][0])):
                    for n in range(len(intersections_horz[k][j][0])):
                        if k!=i and np.abs(intersections_horz[i][j][0][m]-intersections_horz[k][j][0][n]) < l_n:
                            nbrs_horz_isect[m].append(k)
                #for m in range(np.minimum(np.size(intersections_horz[i][j][0]),np.size(intersections_horz[k][j][0]))):
                #    if k != i and np.abs(intersections_horz[i][j][0][m]-intersections_horz[k][j][0][m]) < l_n:
                #        nbrs_horz_isect[m].append(k)
            nbrs_strline_horz.append(nbrs_horz_isect)
        nbrs_horz.append(nbrs_strline_horz)
        if not np.mod(i+1-n0,np.floor((nend-n0)/4).astype(int)):
            print('Process {}: Found neighbors for {} of {} strainlines'.format(pn,i+1-n0,nend-n0))
    q.put(nbrs_vert)
    q.put(nbrs_horz)

def findLCSs(strainlines,nbrs_horz,nbrs_vert,num_horz,num_vert):
    LCSindxs = []
    #progressbar = FloatProgress(min=0,max=len(strainlines))
    #display(progressbar)
    #lmbd2_avgs = [strainline.tailcut_avg_lmbd2() for strainline in strainlines]
    lmbd2_avgs = [strainline.avg_lmbd2() for strainline in strainlines]
    #lngths = [strainline.tailcut_lngth() for strainline in strainlines]
    lngths = [strainline.lngth() for strainline in strainlines]
    for i, (strainline,lmbd2_avg,lngth) in enumerate(zip(strainlines,lmbd2_avgs,lngths)):
        #if not np.mod(i+1,10):
        #    progressbar.value+=10
        for j in range(num_vert):
            for k in range(len(nbrs_vert[i][j])):
                nbr_indxs = set(nbrs_vert[i][j][k])
                T = [lmbd2_avgs[q] for q in nbr_indxs]
                if np.size(T) > 0 and lmbd2_avg >= np.amax(T) and lngth >= strainline.l_min:
                    LCSindxs.append(i)

        for j in range(num_horz):
            for k in range(len(nbrs_horz[i][j])):
                nbr_indxs = set(nbrs_horz[i][j][k])
                T = [lmbd2_avgs[q] for q in nbr_indxs]
                #if (np.size(T) > 0 and lmbd2_avg >= np.amax(T) and strainline.long_enough()):
                if np.size(T) > 0 and lmbd2_avg >= np.amax(T) and lngth >= strainline.l_min:
                    LCSindxs.append(i)
    LCSindxs = list(set(LCSindxs))
    LCSs = []
    for ind in LCSindxs:
        LCSs.append(strainlines[ind])
        print('Strainline {} is an LCS!'.format(ind))
    return LCSs

num_vert_comp = 2
num_horz_comp = 2

vert_x = np.array([0.15,1.05])
horz_y = np.array([0.05,0.95])

isect_horz,isect_vert = find_intersections(strainlines,x_min,x_max,y_min,y_max,num_horz_comp,num_vert_comp,vert_x,horz_y)

l_n = 0.2

#div = np.floor(np.linspace(0,len(strainlines),5)).astype(int)

#for j in range(5):

nbrs_vert,nbrs_horz = find_neighbors(len(strainlines),num_horz_comp,num_vert_comp,isect_horz,isect_vert,l_n,n_proc=16)

LCSs = findLCSs(strainlines,nbrs_horz,nbrs_vert,num_horz_comp,num_vert_comp)

ensure_path_exists('precomputed_lcs/{}'.format(integrator.__name__))
if integrator.__name__ in fixed_step_integrators:
    np.save('precomputed_lcs/{}/lcs_h={}_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.npy'.format(integrator.__name__,h,max_iter,stride,l_f_max,l_min,tol_alpha),LCS)
else:
    np.save('precomputed_lcs/{}/lcs_atol={}_rtol={}_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.npy'.format(integrator.__name__,atol,rtol,max_iter,stride,l_f_max,l_min,tol_alpha),LCS)

