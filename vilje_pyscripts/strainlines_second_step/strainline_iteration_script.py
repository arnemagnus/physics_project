import sys
sys.path.insert(0,'../')
from numba import njit
import numpy as np
import time
from scipy.interpolate import RectBivariateSpline as RBSpline
import os
import errno
import multiprocessing as mp

from numerical_integrators.singlestep import euler,rk2,rk3,rk4
from numerical_integrators.adaptive_step import rkbs32,rkbs54,rkdp54,rkdp87

#import pickle

def ensure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def padded_grid_of_particles(nx,ny,x_min,x_max,y_min,y_max):
    x_0,dx = np.linspace(x_min,x_max,nx,retstep=True)
    y_0,dy = np.linspace(y_min,y_max,ny,retstep=True)

    x = np.empty(nx+4)
    x[0:2] = x_min-2*dx,x_min-dx
    x[2:-2] = x_0
    x[-2:] = x_max+dx,x_max+2*dx

    y = np.empty(ny+4)
    y[0:2] = y_min-2*dy,y_min-dy
    y[2:-2] = y_0
    y[-2:] = y_max+dy,y_max+2*dy

    nx_ = nx+4
    ny_ = ny+4

    grid = np.empty((2,nx_*ny_))

    for j in range(nx_):
        grid[0,j*ny_:(j+1)*ny_] = x[j]
        grid[1,j*ny_:(j+1)*ny_] = y

    return grid

def _find_strain_tensors(grid,nx,ny,dx_m,dy_m,dx_a,dy_a):
    grid_m = grid[0].reshape((2,nx+4,ny+4))
    grid_r = grid[1].reshape((2,nx+4,ny+4))
    grid_t = grid[2].reshape((2,nx+4,ny+4))
    grid_l = grid[3].reshape((2,nx+4,ny+4))
    grid_b = grid[4].reshape((2,nx+4,ny+4))

    df_m = np.empty((nx+4,ny+4,2,2))
    df_a = np.empty((nx+4,ny+4,2,2))

    df_a[:,:,0,0] = (grid_r[0]-grid_l[0])/(2*dx_a)
    df_a[:,:,0,1] = (grid_t[0]-grid_b[0])/(2*dy_a)
    df_a[:,:,1,0] = (grid_r[1]-grid_l[1])/(2*dx_a)
    df_a[:,:,1,1] = (grid_t[1]-grid_b[1])/(2*dy_a)

    df_m[1:-1,:,0,0] = (grid_m[0,2:,:]-grid_m[0,:-2,:])/(2*dx_m)
    df_m[0,:,0,0] = (-3*grid_m[0,0,:]+4*grid_m[0,1,:]-2*grid_m[0,2,:])/(2*dx_m)
    df_m[-1,:,0,0] = (3*grid_m[0,-1,:]-4*grid_m[0,-2,:]+2*grid_m[0,-3,:])/(2*dx_m)

    df_m[:,1:-1,0,1] = (grid_m[0,:,2:]-grid_m[0,:,:-2])/(2*dy_m)
    df_m[:,0,0,1] = (-3*grid_m[0,:,0]+4*grid_m[0,:,1]-2*grid_m[0,:,2])/(2*dy_m)
    df_m[:,-1,0,1] = (3*grid_m[0,:,-1]-4*grid_m[0,:,-2]+2*grid_m[0,:,-3])/(2*dy_m)

    df_m[1:-1,:,1,0] = (grid_m[1,2:,:]-grid_m[1,:-2,:])/(2*dx_m)
    df_m[0,:,1,0] = (-3*grid_m[1,0,:]+4*grid_m[1,1,:]-2*grid_m[1,2,:])/(2*dx_m)
    df_m[-1,:,1,0] = (3*grid_m[1,-1,:]-4*grid_m[1,-2,:]+2*grid_m[1,-3,:])/(2*dx_m)

    df_m[:,1:-1,1,1] = (grid_m[1,:,2:]-grid_m[1,:,:-2])/(2*dy_m)
    df_m[:,0,1,1] = (-3*grid_m[1,:,0]+4*grid_m[1,:,1]-2*grid_m[1,:,2])/(2*dy_m)
    df_m[:,-1,1,1] = (3*grid_m[1,:,-1]-4*grid_m[1,:,-2]+2*grid_m[1,:,-3])/(2*dy_m)

    df_m = df_m.reshape(((nx+4)*(ny+4),2,2))
    df_a = df_a.reshape(((nx+4)*(ny+4),2,2))

    c_m = np.matmul(np.transpose(df_m,axes=(0,2,1)),df_m)
    c_a = np.matmul(np.transpose(df_a,axes=(0,2,1)),df_a)

    return c_m,c_a

def _find_characteristics(c_m,c_a):
    vals = np.linalg.eigvalsh(c_m)
    foo, vecs = np.linalg.eigh(c_a)

    lmbd1 = vals[:,0]
    lmbd2 = vals[:,1]

    xi1 = vecs[:,:,0]
    xi2 = vecs[:,:,1]

    return lmbd1,lmbd2,xi1,xi2

def _find_hessian(lmbd_,nx,ny,dx_m,dy_m):
    lmbd = lmbd_.reshape((nx+4,ny+4))

    hessian = np.empty((nx+4,ny+4,2,2))

    dldx = np.empty(lmbd.shape)
    dldy = np.empty(lmbd.shape)

    d2ldxdy = np.empty(lmbd.shape)
    d2ldydx = np.empty(lmbd.shape)

    dldx[0,:] = (-3*lmbd[0,:]+4*lmbd[1,:]-lmbd[2,:])/(2*dx_m)
    dldx[1:-1,:] = (lmbd[2:,:]-lmbd[:-2,:])/(2*dx_m)
    dldx[-1,:] = (3*lmbd[-1,:]-4*lmbd[-2,:]+lmbd[-3,:])/(2*dx_m)

    dldy[:,0] = (-3*lmbd[:,0]+4*lmbd[:,1]-lmbd[:,2])/(2*dy_m)
    dldy[:,1:-1] = (lmbd[:,2:]-lmbd[:,:-2])/(2*dy_m)
    dldy[:,-1] = (3*lmbd[:,-1]-4*lmbd[:,-2]+lmbd[:,-3])/(2*dy_m)

    d2ldxdy[0,:] = (-3*dldy[0,:]+4*dldy[1,:]-dldy[2,:])/(2*dx_m)
    d2ldxdy[1:-1,:] = (dldy[2:,:]-dldy[:-2,:])/(2*dx_m)
    d2ldxdy[-1,:] = (3*dldy[-1,:]-4*dldy[-2,:]+dldy[-3,:])/(2*dx_m)

    d2ldydx[:,0] = (-3*dldx[:,0]+4*dldx[:,1]-dldx[:,2])/(2*dy_m)
    d2ldydx[:,1:-1] = (dldx[:,2:]-dldx[:,:-2])/(2*dy_m)
    d2ldydx[:,-1] = (3*dldx[:,-1]-4*dldx[:,-2]+dldx[:,-3])/(2*dy_m)

    hessian[:,:,0,1] = d2ldxdy
    hessian[:,:,1,0] = d2ldydx

    hessian[0,:,0,0] = (2*lmbd[0,:]-5*lmbd[1,:]+4*lmbd[2,:]-lmbd[3,:])/(dx_m**2)
    hessian[1:-1,:,0,0] = (lmbd[2:,:]-2*lmbd[1:-1,:]+lmbd[:-2,:])/(dx_m**2)
    hessian[-1,:,0,0] = (2*lmbd[-1,:]-5*lmbd[-2,:]+4*lmbd[-3,:]-lmbd[-4,:])/(dx_m**2)

    hessian[:,0,1,1] = (2*lmbd[:,0]-5*lmbd[:,1]+4*lmbd[:,2]-lmbd[:,3])/(dy_m**2)
    hessian[:,1:-1,1,1] = (lmbd[:,2:]-2*lmbd[:,1:-1]+lmbd[:,:-2])/(dy_m**2)
    hessian[:,-1,1,1] = (2*lmbd[:,-1]-5*lmbd[:,-2]+4*lmbd[:,-3]-lmbd[:,-4])/(dy_m**2)

    return hessian.reshape(((nx+4)*(ny+4),2,2))

def characteristics(integrator,fixed_step_integrators,stride,atol,rtol,nx,ny,x_min,x_max,y_min,y_max,t_start=0.0,t_end=20.0):
    dx_m = (x_max-x_min)/(nx-1)
    dy_m = (y_max-y_min)/(ny-1)
    dx_a = np.minimum(1e-05,dx_m*1e-2)
    dy_a = np.minimum(1e-05,dy_m*1e-2)

    if integrator.__name__ in fixed_step_integrators:
        grid_fin = np.load('../advection_first_step/precomputed_advections/{}/advected_grid_Nx={}_Ny={}_dx_main={}_dy_main={}_dx_aux={}_dy_aux={}_t_start={}_t_end={}_h={}.npy'.format(integrator.__name__,nx,ny,dx_m,dy_m,dx_a,dy_a,t_start,t_end,stride))
    else:
        grid_fin = np.load('../advection_first_step/precomputed_advections/{}/advected_grid_Nx={}_Ny={}_dx_main={}_dy_main={}_dx_aux={}_dy_aux={}_t_start={}_t_end={}_atol={}_rtol={}.npy'.format(integrator.__name__,nx,ny,dx_m,dy_m,dx_a,dy_a,t_start,t_end,atol,rtol))

    tens_main,tens_aux = _find_strain_tensors(grid_fin,nx,ny,dx_m,dy_m,dx_a,dy_a)

    lmbd1,lmbd2,xi1,xi2 = _find_characteristics(tens_main,tens_aux)

    hess_lmbd2 = _find_hessian(lmbd2,nx,ny,dx_m,dy_m)
    return lmbd1,lmbd2,hess_lmbd2,xi1,xi2


def find_ab_mask(lmbd1,lmbd2,hess_lmbd2,xi2):
    return np.logical_and(_a_true(lmbd1,lmbd2),_b_true(hess_lmbd2,xi2))

def _a_true(lmbd1,lmbd2):
    return np.logical_and(np.less(lmbd1,lmbd2),np.greater(lmbd2,1))

def _b_true(hess_lmbd2,xi2):
    return np.less_equal(np.sum(xi2*np.sum(hess_lmbd2*xi2[...,np.newaxis],axis=1),axis=1),0)

def find_partial_g0_mask(nx,ny,num_horz,num_vert):
    mask = np.zeros((nx,ny),dtype=np.bool)
    stride_horz = np.floor(nx/(num_horz+1)).astype(int)
    stride_vert = np.floor(ny/(num_vert+1)).astype(int)

    for j in range(1,num_vert+1):
        mask[np.minimum(j*stride_horz,nx-1),:] = True
    for j in range(1,num_horz+1):
        mask[:,np.minimum(j*stride_vert,ny-1)] = True

    return mask.reshape(nx*ny)

class LinearSpecialDerivative:
    def __init__(self,grid,xi):
        self.grid = grid
        self.x_min = self.grid[0,0,0]
        self.x_max = self.grid[0,-1,0]
        self.y_min = self.grid[1,0,0]
        self.y_max = self.grid[1,0,-1]
        self.dx = self.grid[0,1,0]-self.grid[0,0,0]
        self.dy = self.grid[1,0,1]-self.grid[1,0,0]
        self.xi = xi
        self.prev = None
    def clear_previous(self):
        self.prev = None
    def set_previous(self,prev):
        self.prev = prev
    def __call__(self,pos):
        pos[0] = np.maximum(self.x_min-self.dx,np.minimum(pos[0],self.x_max+self.dx))
        pos[1] = np.maximum(self.y_min-self.dy,np.minimum(pos[1],self.y_max+self.dy))

        i = np.maximum(0,np.minimum(pos[0]/self.dx,self.grid.shape[1]-2)).astype(int)
        j = np.maximum(0,np.minimum(pos[1]/self.dy,self.grid.shape[2]-2)).astype(int)

        subxi = self.xi[i:i+2,j:j+2]

        for ii in range(2):
            for jj in range(2):
                if ii == 0 and jj == 0:
                    pass
                else:
                    dps = np.sign(np.dot(subxi[0,0],subxi[ii,jj]))
                    if dps < 0:
                        subxi[ii,jj] = -subxi[ii,jj]

        wr = (pos[0]-i*self.dx)/self.dx
        wl = 1-wr
        wt = (pos[1]-j*self.dy)/self.dy
        wb = 1-wt

        xi = wl*(wb*subxi[0,0]+wt*subxi[0,1]) + wr*(wb*subxi[1,0]+wt*subxi[1,1])
        xi = xi/np.sqrt(xi[0]**2+xi[1]**2)

        if self.prev is None:
            sign = 1
        else:
            sign = -1 if np.sign(np.dot(self.prev,xi)) < 0 else 1

        return sign*xi

class InABDomain:
    def __init__(self,pos_init,lmbd1,lmbd2,hess_lmbd2,xi2,x_min,x_max,y_min,y_max,padding_factor=0.01):
        self._lmbd1_spline = RBSpline(pos_init[1,0,:],pos_init[0,:,0],lmbd1.T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        self._lmbd2_spline = RBSpline(pos_init[1,0,:],pos_init[0,:,0],lmbd2.T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        self._hess_lmbd2_xx_spline = RBSpline(pos_init[1,0,:],pos_init[0,:,0],hess_lmbd2[:,:,0,0].T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        self._hess_lmbd2_xy_spline = RBSpline(pos_init[1,0,:],pos_init[0,:,0],hess_lmbd2[:,:,0,1].T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        self._hess_lmbd2_yx_spline = RBSpline(pos_init[1,0,:],pos_init[0,:,0],hess_lmbd2[:,:,1,0].T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        self._hess_lmbd2_yy_spline = RBSpline(pos_init[1,0,:],pos_init[0,:,0],hess_lmbd2[:,:,1,1].T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        self._xi2_x_spline = RBSpline(pos_init[1,0,:],pos_init[0,:,0],xi2[:,:,0].T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
        self._xi2_y_spline = RBSpline(pos_init[1,0,:],pos_init[0,:,0],xi2[:,:,1].T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)

    def __call__(self,pos):
        lmbd1 = self._lmbd1_spline.ev(pos[1],pos[0])
        lmbd2 = self._lmbd2_spline.ev(pos[1],pos[0])
        hess = np.array([[self._hess_lmbd2_xx_spline.ev(pos[1],pos[0]),self._hess_lmbd2_xy_spline.ev(pos[1],pos[0])],[self._hess_lmbd2_yx_spline.ev(pos[1],pos[0]),self._hess_lmbd2_yy_spline.ev(pos[1],pos[0])]])
        xi2 = np.array([self._xi2_x_spline.ev(pos[1],pos[0]),self._xi2_y_spline.ev(pos[1],pos[0])])
        #xi2 = xi2/np.sqrt(xi2[0]**2+xi2[1]**2)
        return ((lmbd1 != lmbd2) and (lmbd2 > 1) and np.less_equal(np.dot(xi2,np.dot(hess,xi2)),0))

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
    rhs_f.clear_previous()
    rhs_f.set_previous(rhs_f(strainline_f.pos[:,-1]))

    rhs_b.clear_previous()
    rhs_b.set_previous(-rhs_f.prev)


    tailcut_index = 0

    pos_trial = RK4_iterator(strainline_f.pos[:,-1],stride,rhs_f)
    while l_f<l_f_max and alpha(pos_trial)>tol_alpha and in_domain(pos_trial) and counter<max_iter:
        if not in_ab(pos_trial):
            l_f += np.sqrt((pos_trial[0]-strainline_f.pos[0,-1])**2+(pos_trial[1]-strainline_f.pos[1,-1])**2)
        else:
            l_f = 0.
            tailcut_index = np.shape(strainline_f.pos)[1]
        strainline_f.append(pos_trial)
        rhs_f.set_previous(rhs_f(strainline_f.pos[:,-2]))
        counter+=1
        pos_trial = RK4_iterator(strainline_f.pos[:,-1],stride,rhs_f)

    if alpha(strainline_f.pos[:,-1])<=tol_alpha:
        strainline_f.stationary[1] = True
    if not in_domain(pos_trial):
        strainline_f.outs_dom[1] = True
    if counter == max_iter:
        strainline_f.max_iter[1] = True
    if l_f>=l_f_max:
        strainline_f.cont_failure[1] = True
        strainline_f.tailcut_end = tailcut_index
    else:
        strainline_f.tailcut_end = strainline_f.pos.shape[1]-1



    strainline_b = Strainline(startpoint,l_min,l_f_max,lmbd2_spline)
    counter = 0
    l_f = 0.
    L = 0.
    tailcut_index = 0

    pos_trial = RK4_iterator(strainline_b.pos[:,-1],stride,rhs_b)
    while l_f<l_f_max and alpha(pos_trial)>tol_alpha and in_domain(pos_trial) and counter<max_iter:
        if not in_ab(pos_trial):
            l_f += np.sqrt((pos_trial[0]-strainline_b.pos[0,-1])**2+(pos_trial[1]-strainline_b.pos[1,-1])**2)
        else:
            l_f = 0.
            tailcut_index = np.shape(strainline_b.pos)[1]
        strainline_b.append(pos_trial)
        rhs_b.set_previous(rhs_b(strainline_b.pos[:,-2]))
        counter+=1
        pos_trial = RK4_iterator(strainline_b.pos[:,-1],stride,rhs_b)


    if alpha(strainline_b.pos[:,-1])<=tol_alpha:
        strainline_b.stationary[0] = True
    if not in_domain(pos_trial):
        strainline_b.outs_dom[0] = True
    if counter == max_iter:
        strainline_b.max_iter[0] = True
    if l_f>=l_f_max:
        strainline_b.cont_failure[0] = True
        strainline_b.tailcut_start = strainline_b.pos.shape[1]-tailcut_index
    else:
        strainline_b.tailcut_start = 0


    strainline_b.len = np.sum(np.sqrt((np.diff(strainline_b.pos,axis=1)**2).sum(axis=0)))



    strainline_b.startpoint_index = strainline_b.pos.shape[1]-1
    strainline_b.pos = np.hstack((strainline_b.pos[:,::-1],strainline_f.pos[:,1:]))
    strainline_b.stationary[1] = strainline_f.stationary[1]
    strainline_b.cont_failure[1] = strainline_f.cont_failure[1]
    strainline_b.outs_dom[1] = strainline_f.outs_dom[1]
    strainline_b.max_iter[1] = strainline_f.max_iter[1]
    strainline_b.tailcut_end = strainline_b.startpoint_index + strainline_f.tailcut_end - 1

    return strainline_b

def computestrainlines(dompts,max_iter,rhs_f,rhs_b,stride,l_f_max,l_min,alpha,tol_alpha,in_ab,in_domain,lmbd2_spline,q):
    strainlines = []
    for j in range(dompts.shape[1]):
        strainlines.append(iteratestrainline(dompts[:,j],max_iter,rhs_f,rhs_b,stride,l_f_max,l_min,alpha,tol_alpha,in_ab,in_domain,lmbd2_spline))
    q.put(strainlines)

def advect_strainlines(integrator,fixed_step_integrators,h,atol,rtol,nx=1000,ny=500,x_min=0.,x_max=2.,y_min=0.,y_max=1.,t_start=0.,t_end=20.):
    lmbd1_,lmbd2_,hess_lmbd2_,xi1_,xi2_ = characteristics(integrator,fixed_step_integrators,h,atol,rtol,nx,ny,x_min,x_max,y_min,y_max)


    grid_ = padded_grid_of_particles(nx,ny,x_min,x_max,y_min,y_max)
    x_ = grid_[0,::ny+4]
    y_ = grid_[1,:ny+4]
    x = x_[2:-2]
    y = y_[2:-2]

    _inner_mask = np.zeros((nx+4,ny+4),dtype=np.bool)
    _inner_mask[2:-2,2:-2] = True
    _inner_mask = _inner_mask.reshape((nx+4)*(ny+4))

    grid = grid_[:,_inner_mask]
    lmbd1 = lmbd1_[_inner_mask]
    lmbd2 = lmbd2_[_inner_mask]
    hess_lmbd2 = hess_lmbd2_[_inner_mask,:,:]
    xi1 = xi1_[_inner_mask,:]
    xi2 = xi2_[_inner_mask,:]

    mask_ab = find_ab_mask(lmbd1,lmbd2,hess_lmbd2,xi2)

    num_horz_g0 = 4
    num_vert_g0 = 4

    g0 = grid[:,np.logical_and(mask_ab,find_partial_g0_mask(nx,ny,num_horz_g0,num_vert_g0))]

    slines = []

    max_iter = 10000
    stride = 0.001
    l_f_max = 0.2
    l_min = 1.
    tol_alpha = 1.e-6

    padding_factor = 0.01

    inAB = InABDomain(grid.reshape(2,nx,ny),lmbd1.reshape(nx,ny),lmbd2.reshape(nx,ny),hess_lmbd2.reshape(nx,ny,2,2),xi2.reshape(nx,ny,2),x_min,x_max,y_min,y_max,padding_factor)

    in_num_dom = InNumericalDomain(x_min,x_max,y_min,y_max,nx,ny)

    lmbd1_spline = RBSpline(y,x,lmbd1.reshape(nx,ny).T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
    lmbd2_spline = RBSpline(y,x,lmbd2.reshape(nx,ny).T,bbox=[y_min-(y_max-y_min)*padding_factor,y_max+(y_max-y_min)*padding_factor,x_min-(x_max-x_min)*padding_factor,x_max+(x_max-x_min)*padding_factor],kx=1,ky=1)
    alpha = Alpha(lmbd1_spline,lmbd2_spline)

    rhs_f = LinearSpecialDerivative(grid.reshape(2,nx,ny),xi1.reshape(nx,ny,2))
    rhs_b = LinearSpecialDerivative(grid.reshape(2,nx,ny),xi1.reshape(nx,ny,2))

    nproc = 16

    print(g0.shape)

    div = np.floor(np.linspace(0,g0.shape[1],nproc+1)).astype(int)

    qs = [mp.Queue() for j in range(nproc)]
    ps = [mp.Process(target = computestrainlines,
                    args=(g0[:,div[j]:div[j+1]],max_iter,rhs_f,rhs_b,stride,l_f_max,l_min,alpha,tol_alpha,inAB,in_num_dom,lmbd2_spline,qs[j]))
            for j in range(nproc)]
    for p in ps:
        p.start()
    for j,q in enumerate(qs):
        slines += q.get()
    for p in ps:
        p.join()


    ensure_path_exists('precomputed_strainlines/{}'.format(integrator.__name__))
    if integrator.__name__ in fixed_step_integrators:
#        with open('precomputed_strainlines/{}/strainlines_h={}_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.pkl'.format(integrator.__name__,h,max_iter,stride,l_f_max,l_min,tol_alpha),'wb') as output:
#            pickle.dump(slines,output,pickle.HIGHEST_PROTOCOL)
        np.save('precomputed_strainlines/{}/strainlines_h={}_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.npy'.format(integrator.__name__,h,max_iter,stride,l_f_max,l_min,tol_alpha),slines)
    else:
        #with open('precomputed_strainlines/{}/strainlines_atol={}_rtol={}_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.pkl'.format(integrator.__name__,atol,rtol,max_iter,stride,l_f_max,l_min,tol_alpha),'wb') as output:
        #    pickle.dump(slines,output,pickle.HIGHEST_PROTOCOL)
        np.save('precomputed_strainlines/{}/strainlines_atol={}_rtol={}_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.npy'.format(integrator.__name__,atol,rtol,max_iter,stride,l_f_max,l_min,tol_alpha),slines)

    #if rank == 0:
    #    slines[div[rank]:div[rank+1]] = computestrainlines(g0[:,div[rank]:div[rank+1]],max_iter,rhs_f,rhs_b,stride,l_f_max,l_min,alpha,tol_alpha,inAB,in_num_dom,lmbd2_spline)
    #    for j in range(1,size):
    #        comm.recv(slines[div[rank]:div[rank+1]],j)
    #    ensure_path_exists('precomuted_strainlines/{}'.format(integrator.__name__))
    #    if integrator.__name__ in fixed_step_integrators:
    #        np.save('precomputed_strainlines/{}/strainlines_h={}_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.npy'.format(h,max_iter,stride,l_f_max,l_min,tol_alpha),strainlines)
    #    else:
    #        np.save('precomputed_strainlines/{}/strainlines_atol={}_rtol={}_max_iter={}_stride={}_l_f_max={}_l_min={}_tol_alpha={}.npy'.format(atol,rtol,max_iter,stride,l_f_max,l_min,tol_alpha),strainlines)
    #else:
    #    comm.send(computestrainlines(g0[:,div[rank]:div[rank+1]],max_iter,rhs_f,rhs_b,stride,l_f_max,l_min,alpha,tol_alpha,inAB,in_num_dom,lmbd2_spline),dest=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mt',help='Name of integrator used for advecting tracers',type=str)
    parser.add_argument('st',help='(Initial) integration stride used',type=np.float64)
    parser.add_argument('at',help='Absolute tolerance used (adaptive methods)',type=np.float64)
    parser.add_argument('rt',help='Relative tolerance used (adaptive methods)',type=np.float64)
    args = parser.parse_args()

    fixed_step_integrators = set([euler.__name__,rk2.__name__,rk3.__name__,rk4.__name__])
    integrators = [euler,rk2,rk3,rk4,rkbs32,rkbs54,rkdp54,rkdp87]
    for integ in integrators:
        if integ.__name__ == args.mt:
            integrator = integ
            break

    advect_strainlines(integrator,fixed_step_integrators,args.st,args.at,args.rt)
