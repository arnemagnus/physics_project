# Numba (JiT)
from numba import njit

# MPI
from mpi4py import MPI

# Numpy
import numpy as np

import multiprocessing as mp

# (Primitive) timing functionality
import time

# Spline interpolation:
from scipy.interpolate import RectBivariateSpline

# Check whether folders exist or not, necessary
# for storing advected states:
import os
import errno

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create output directory for precomputed characteristics, if it does not already exist:
#ensure_path_exists('precomputed_strainlines/rk4')



x_min,x_max = 0,2
y_min,y_max = 0,1

nx,ny = 1000,500

grid = np.load('precomputed_characteristics/rk4/grid_nx=1000_ny=500.npy')
lmbd1 = np.load('precomputed_characteristics/rk4/lmbd1_nx=1000_ny=500_t_start=0._t_end=20._h=0.1.npy')
lmbd2 = np.load('precomputed_characteristics/rk4/lmbd2_nx=1000_ny=500_t_start=0._t_end=20._h=0.1.npy')
lapl_lmbd2 = np.load('precomputed_characteristics/rk4/lapl_lmbd2_nx=1000_ny=500_t_start=0._t_end=20._h=0.1.npy')
xi1 = np.load('precomputed_characteristics/rk4/xi1_nx=1000_ny=500_t_start=0._t_end=20._h=0.1.npy')
xi2 = np.load('precomputed_characteristics/rk4/xi2_nx=1000_ny=500_t_start=0._t_end=20._h=0.1.npy')
lapl_xi2 = np.load('precomputed_characteristics/rk4/lapl_xi2_nx=1000_ny=500_t_start=0._t_end=20._h=0.1.npy')
g0 = np.load('precomputed_characteristics/rk4/g0_nx_1000_ny=500.npy')

class LinearSpecialDerivative:
    def __init__(self,grid,xi):
        self.grid = grid
        self.dx = self.grid[0,1,0]-self.grid[0,0,0]
        self.dy = self.grid[1,0,1]-self.grid[1,0,0]
        self.xi = xi
        self.prev = None
        self.reverse = False
    def clear_previous(self):
        self.prev = None
    def set_previous(self,prev):
        self.prev = prev
    def flip(self):
        self.reverse = True
    def flip_back(self):
        self.reverse = False
    def __call__(self,pos):
        
        if pos[0] > x_max + self.dx:
            pos[0] = x_max + self.dx
        elif pos[0] < x_min - self.dx:
            pos[0] = x_min - self.dx
        
        if pos[1] > y_max + self.dy:
            pos[1] = y_max + self.dy
        elif pos[1] < y_min - self.dy:
            pos[1] = y_min - self.dy
        
        i = int(np.maximum(0,np.minimum(pos[0]/self.dx,np.size(self.grid,1)-2)))
        j = int(np.maximum(0,np.minimum(pos[1]/self.dy,np.size(self.grid,2)-2)))

        
        subxi = self.xi[:,i:i+2,j:j+2]

        # Choose pivot vector and fix directions of others based on this one
        for ii in range(2):
            for jj in range(2):
                if ii == 0 and jj == 0:
                    pass
                else:
                    dp = np.sign(np.dot(subxi[:,0,0],subxi[:,ii,jj]))
                    if dp < 0:
                        subxi[:,ii,jj] = - subxi[:,ii,jj]
                    #subxi[:,ii,jj] = np.sign(np.dot(subxi[:,0,0],subxi[:,ii,jj])) * subxi[:,ii,jj]
                    
        wr = (pos[0]-i*self.dx)/self.dx
        wl = 1-wr
        wt = (pos[1]-j*self.dy)/self.dy
        wb = 1-wt
        
        xi = wl*(wb*subxi[:,0,0]+wt*subxi[:,0,1])+wr*(wb*subxi[:,1,0]+wt*subxi[:,1,1])
        # Normalize xi
        xi = xi / np.sqrt(xi[0]**2+xi[1]**2)
        
        if self.prev is None:
            if not self.reverse:
                sign = 1.
            else:
                sign = -1.
        else:
            sign = -1. if np.sign(np.dot(self.prev,xi)) < 0 else 1.
        return sign*xi
    
class InABDomain:
    def __init__(self,pos_init,lmbd1,lmbd2,lapl_lmbd2,xi2,x_min,x_max,y_min,y_max,padding_factor=0.01):
        self._lmbd1_spline = RectBivariateSpline(pos_init[1,0,:],pos_init[0,:,0],lmbd1.T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        self._lmbd2_spline = RectBivariateSpline(pos_init[1,0,:],pos_init[0,:,0],lmbd2.T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        self._lapl_lmbd2_spline = RectBivariateSpline(pos_init[1,0,:],pos_init[0,:,0],lapl_lmbd2.T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        self._xi2_x_spline = RectBivariateSpline(pos_init[1,0,:],pos_init[0,:,0],xi2[0].T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        self._xi2_y_spline = RectBivariateSpline(pos_init[1,0,:],pos_init[0,:,0],xi2[1].T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        
    def __call__(self,pos):
        lmbd1 = self._lmbd1_spline.ev(pos[1],pos[0])
        lmbd2 = self._lmbd2_spline.ev(pos[1],pos[0])
        lapl_lmbd2 = self._lapl_lmbd2_spline.ev(pos[1],pos[0])
        xi2 = np.array([self._xi2_x_spline.ev(pos[1],pos[0]),self._xi2_y_spline.ev(pos[1],pos[0])])
        xi2 = xi2/np.sqrt(xi2[0]**2+xi2[1]**2)
        return ((lmbd1 is not lmbd2) and (lmbd2 > 1) and (np.dot(xi2,lapl_lmbd2*xi2) <= 0))
    
class InNumericalDomain:
    def __init__(self,x_min,x_max,y_min,y_max,nx,ny):
        dx = (x_max-x_min)/(nx-1)
        dy = (y_max-y_min)/(ny-1)
        self._x_min = x_min-dx
        self._x_max = x_max+dx
        self._y_min = y_min-dy
        self._y_max = y_max+dx
    
    def __call__(self,pos):
        return pos[0] >= self._x_min and pos[0] <= self._x_max and pos[1] >= self._y_min and pos[1] <= self._y_max
    
class Alpha:
    def __init__(self,lmbd1_spline,lmbd2_spline):
        self.lmbd1_spline = lmbd1_spline
        self.lmbd2_spline = lmbd2_spline
    def __call__(self,pos):
        lmbd1 = self.lmbd1_spline.ev(pos[1],pos[0])
        lmbd2 = self.lmbd2_spline.ev(pos[1],pos[0])
        return ((lmbd2-lmbd1)/(lmbd2+lmbd1))**2

class Strainline:
    def __init__(self,startpoint,l_min,l_f_max,lmbd2_spline):
        self.pos = np.array([startpoint]).reshape((2,1))
        self.l_min = l_min
        self.len = 0.
        self.lmbd2_spline = lmbd2_spline
        self.stationary = np.zeros(2,dtype=np.bool)
        self.cont_failure = np.zeros(2,dtype=np.bool)
        self.outs_dom = np.zeros(2,dtype=np.bool)
        self.max_iter = np.zeros(2,dtype=np.bool)
        self.startpoint_index = 0
        #self.tail_start = 0
        #self.tail_end = 0
    def long_enough(self):
        return self.len >= self.l_min
    def append(self,pos):
        self.pos = np.hstack((self.pos,pos.reshape((2,1))))
    def avg_lmbd2(self):
        return np.mean(self.lmbd2_spline.ev(self.pos[1,:],self.pos[0,:]))
    def set_length(self,l):
        self.len = l

def RK4_iterator(pos_prev,stride,rhs):
    pos_new = np.zeros(2)
    lk = np.zeros((2,5))
    lk[:,1] = rhs(pos_prev)
    lk[:,2] = rhs(pos_prev+lk[:,1]*stride/2.)
    lk[:,3] = rhs(pos_prev+lk[:,2]*stride/2.)
    lk[:,4] = rhs(pos_prev+lk[:,3]*stride)
    lk[:,0] = (lk[:,1]+2*lk[:,2]+2*lk[:,3]+lk[:,4])/6.
    pos_new[0] = pos_prev[0]+lk[0,0]*stride
    pos_new[1] = pos_prev[1]+lk[1,0]*stride
    return pos_new

def iteratestrainline(startpoint,max_iter,rhs_f,rhs_b,stride,l_f_max,l_min,alpha,tol_alpha,in_ab,in_domain,lmbd2_spline):
    strainline_f = Strainline(startpoint,l_min,l_f_max,lmbd2_spline)
    counter = 0
    l_f = 0.
    L = 0.
    rhs_f.clear_previous()
    rhs_f.set_previous(rhs_f(strainline_f.pos[:,-1]))
    
    rhs_b.clear_previous()
    rhs_b.set_previous(-rhs_f.prev)
    
    
    pos_trial = RK4_iterator(strainline_f.pos[:,-1],stride,rhs_f)
    while l_f<l_f_max and alpha(pos_trial)>tol_alpha and in_domain(pos_trial) and counter<max_iter:
        L += np.sqrt((pos_trial[0]-strainline_f.pos[0,-1])**2+(pos_trial[1]-strainline_f.pos[1,-1])**2)
        if not in_ab(pos_trial):
            l_f += np.sqrt((pos_trial[0]-strainline_f.pos[0,-1])**2+(pos_trial[1]-strainline_f.pos[1,-1])**2)
        else:
            l_f = 0.
        strainline_f.append(pos_trial)
        rhs_f.set_previous(rhs_f(strainline_f.pos[:,-2]))
        counter+=1
        pos_trial = RK4_iterator(strainline_f.pos[:,-1],stride,rhs_f)
    if l_f>=l_f_max:
        strainline_f.cont_failure[1] = True
        ## Cut tail end off in case of continious failure
        #ind = strainline_f.pos.shape[1]-1
        #while not in_ab(strainline_f.pos[:,ind]):
        #    ind-=1
        #strainline_f.tail_end = ind+1
    if alpha(strainline_f.pos[:,-1])<=tol_alpha:
        strainline_f.stationary[1] = True
    if not in_domain(pos_trial):
        strainline_f.outs_dom[1] = True
    if counter == max_iter:
        strainline_f.max_iter[1] = True
        
    #print(strainline_f.pos[:,:10])
        
    strainline_f.set_length(L)
        
    strainline_b = Strainline(startpoint,l_min,l_f_max,lmbd2_spline)
    counter = 0
    l_f = 0.
    L = 0.
    
    pos_trial = RK4_iterator(strainline_b.pos[:,-1],stride,rhs_b)
    while l_f<l_f_max and alpha(pos_trial)>tol_alpha and in_domain(pos_trial) and counter<max_iter:
        L += np.sqrt((pos_trial[0]-strainline_b.pos[0,-1])**2+(pos_trial[1]-strainline_b.pos[1,-1])**2)
        if not in_ab(pos_trial):
            l_f += np.sqrt((pos_trial[0]-strainline_b.pos[0,-1])**2+(pos_trial[1]-strainline_b.pos[1,-1])**2)
        else:
            l_f = 0.
        strainline_b.append(pos_trial)
        rhs_b.set_previous(rhs_b(strainline_b.pos[:,-2]))
        counter+=1
        pos_trial = RK4_iterator(strainline_b.pos[:,-1],stride,rhs_b)
    if l_f>=l_f_max:
        strainline_b.cont_failure[0] = True
        ## Cut tail end off in case of continious failure
        #ind = strainline_b.pos.shape[1]-1
        #while not in_ab(strainline_b.pos[:,ind]):
        #    ind-=1
        #strainline_b.pos = strainline_b.pos[:,:ind+1]
        #strainline_b.tail_begin = strainline_b.pos.shape[1] -2 + ind
    if alpha(strainline_b.pos[:,-1])<=tol_alpha:
        strainline_b.stationary[0] = True
    if not in_domain(pos_trial):
        strainline_b.outs_dom[0] = True
    if counter == max_iter:
        strainline_b.max_iter[0] = True
        
        
    strainline_b.set_length(L)
    
    strainline_b.startpoint_index = strainline_b.pos.shape[1]-1
    #strainline_b.tail_end = strainline_b.pos.shape[1]-1 + strainline_f.tail_end 
    strainline_b.pos = np.hstack((strainline_b.pos[:,::-1],strainline_f.pos[:,1:]))
    strainline_b.len += strainline_f.len
    strainline_b.stationary[1] = strainline_f.stationary[1]
    strainline_b.cont_failure[1] = strainline_f.cont_failure[1]
    strainline_b.outs_dom[1] = strainline_f.outs_dom[1]
    strainline_b.max_iter[1] = strainline_f.max_iter[1]
    
    return strainline_b

def computestrainlines(dompts,max_iter,rhs_f,rhs_b,stride,l_f_max,l_min,alpha,tol_alpha,in_ab,in_domain,lmbd2_spline):
    strainlines = np.empty(np.shape(dompts)[1])
    for j in range(np.shape(dompts)[1]):
        strainlines[j] = (iteratestrainline(dompts[:,j],max_iter,rhs_f,rhs_b,stride,l_f_max,l_min,alpha,tol_alpha,in_ab,in_domain,lmbd2_spline)
        #if not (np.mod(j +  1 +  np.floor(np.size(dompts,1)/4).astype(int), np.floor(np.size(dompts,1)/4).astype(int))):
        #    print('Process {}: Finished integrating strainline candidate {} of {}'.format(pn,j+1,np.size(dompts,1)))
    return strainlines

def develop_strainlines(max_iter,stride,l_f_max,l_min,tol_alpha,padding_factor=0.01):
    
    recv_buf = np.empty(np.shape(g0)[1])

    in_AB_domain = InABDomain(np.reshape(grid,(2,nx,ny)),np.reshape(lmbd1,(nx,ny)),np.reshape(lmbd2,(nx,ny)),np.reshape(lapl_lmbd2,(nx,ny)),np.reshape(xi2,(2,nx,ny)),x_min,x_max,y_min,y_max,padding_factor)
    in_numerical_domain = InNumericalDomain(x_min,x_max,y_min,y_max,nx,ny)

    lmbd1_spline = RectBivariateSpline(np.reshape(grid,(2,nx,ny))[1,0,:],np.reshape(grid,(2,nx,ny))[0,:,0],np.reshape(lmbd1,(nx,ny)).T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
    lmbd2_spline = RectBivariateSpline(np.reshape(grid,(2,nx,ny))[1,0,:],np.reshape(grid,(2,nx,ny))[0,:,0],np.reshape(lmbd2,(nx,ny)).T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)

    rhs_f = LinearSpecialDerivative(np.reshape(grid,(2,nx,ny)),np.reshape(xi1,(2,nx,ny)))
    rhs_b = LinearSpecialDerivative(np.reshape(grid,(2,nx,ny)),np.reshape(xi1,(2,nx,ny)))
    alpha = Alpha(lmbd1_spline,lmbd2_spline)   

    div = np.floor(np.linspace(0,np.shape(g0)[1],size+1)).astype(int)

    if rank == 0:
        strainlines = np.empty(np.shape(g0)[1])
        strainlines[div[rank]:div[rank+1]] = computestrainlines(g0[:,div[rank]:div[rank+1]],max_iter,rhs_f,rhs_b,stride,l_f_max,l_min,alpha,tol_alpha,in_AB_domain,in_numerical_domain,lmbd2_spline)
        for i in range(1,size):
            comm.Recv(recv_buf[div[i]:div[i+1]],i)
            strainlines[div[i]:div[i+1]] = recv_buf[div[i]:div[i+1]]
        np.save('precomputed_strainlines/rk4/nx=1000_ny=500_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.npy'.format(max_iter,stride,l_f_max,l_min,tol_alpha),strainlines)
    else:
        comm.Send(computestrainlines(g0[:,div[rank]:div[rank+1]],max_iter,rhs_f,rhs_b,stride,l_f_max,l_min,alpha,tol_alpha,in_AB_domain,in_numerical_domain,lmbd2_spline),dest=0)
    
    
    #return strainlines

max_iter = 20000
stride = 0.0005
l_f_max = 0.2
l_min = 1.
tol_alpha = 1.e-6

develop_strainlines(max_iter,stride,l_f_max,l_min,tol_alpha)

#np.save('precomputed_strainlines/rk4/nx=1000_ny=500_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.npy'.format(max_iter,stride,l_f_max,l_min,tol_alpha),strainlines)
    
