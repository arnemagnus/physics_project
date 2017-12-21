import sys
sys.path.insert(0,'../')
from mpi4py import MPI
from numba import njit
import numpy as np
import time
import os
import errno

from numerical_integrators.singlestep import euler,rk2,rk3,rk4
from numerical_integrators.adaptive_step import rkbs32,rkbs54,rkdp54,rkdp87

def ensure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

@njit
def doublegyre_wrapper(t,x):
    A = 0.1
    e = 0.1
    w = 2*np.pi/10
    return ___doublegyre(t,x,A,e,w)

@njit
def ___doublegyre(t,x,A,e,w):
    a = e*np.sin(w*t)
    b = 1-2*e*np.sin(w*t)
    f = a*x[0,:]**2 + b*x[0,:]
    dfdx = 2*a*x[0,:] + b

    v = np.empty(x.shape)
    v[0,:] = -np.pi*A*np.sin(np.pi*f)*np.cos(np.pi*x[1,:])
    v[1,:] = np.pi*A*np.cos(np.pi*f)*np.sin(np.pi*x[1,:])*dfdx
    return v

def endpoints_fixed(t_start,t_end,pos_start,stride,integrator,rhs):
    pos_fin = np.copy(pos_start)
    if t_end>t_start:
        for j in range(pos_fin.shape[0]):
            n_steps = np.ceil((t_end-t_start)/stride).astype(int)
            h = (t_end-t_start)/n_steps
            for i in range(n_steps):
                foo, pos_fin[j], bar = integrator(t_start+i*h, pos_fin[j], h, rhs)
    else:
        for j in range(np.size(pos_fin,0)):
            n_steps = np.ceil((t_start-t_end)/stride).astype(int)
            h = -(t_start-t_end)/n_steps
            for i in range(n_steps):
                foo, pos_fin[j], bar = integrator(t_start+i*h, pos_fin[j], h, rhs)
    return pos_fin

def endpoints_adaptive(t_start,t_end,pos_start,stride,integrator,rhs,atol,rtol):
    pos_fin = np.copy(pos_start)
    n_calls = np.zeros((5,pos_fin.shape[2]),dtype=int)
    if t_end>t_start:
        for j in range(np.size(pos_fin,0)):
            t = np.ones(np.shape(pos_fin[j])[1])*t_start
            h = np.ones(np.shape(pos_fin[j])[1])*stride
            while np.any(np.less(t,t_end)):
                mask = np.less(t,t_end)
                h[mask] = np.minimum(h[mask],t_end-t[mask])
                t[mask],pos_fin[j][:,mask],h[mask] = integrator(t[mask],pos_fin[j][:,mask],h[mask],rhs,atol,rtol)
                n_calls[j,mask]+=1
    else:
        for j in range(np.size(pos_fin,0)):
            t = np.ones(np.shape(pos_fin[j])[1])*t_start
            h = np.ones(np.shape(pos_fin[j])[1])*stride
            while np.any(t>t_end):
                mask = np.greater(t,t_end)
                h[mask] = np.sign(h[mask])*np.minimum(np.abs(h[mask]),np.abs(t[mask]-t_end))
                t[mask],pos_fin[j][:,mask],h[mask] = integrator(t[mask],pos_fin[j][:,mask],h[mask],rhs,atol,rtol)
                n_calls[j,mask]+=1
    return pos_fin,n_calls

def padded_grid_of_particles(nx,ny,x_min,x_max,y_min,y_max):
    x_0,dx = np.linspace(x_min,x_max,nx,retstep=True)
    y_0,dy = np.linspace(y_min,y_max,ny,retstep=True)

    x = np.empty(nx+4)
    x[0:2] = -2*dx, -dx
    x[2:-2] = x_0
    x[-2:] = x_max+dx, x_max+2*dx

    y = np.empty(ny+4)
    y[0:2] = -2*dy, -dy
    y[2:-2] = y_0
    y[-2:] = y_max+dy, y_max+2*dy

    nx_ = nx+4
    ny_ = ny+4

    grid = np.empty((2,nx_*ny_))

    for j in range(nx_):
        grid[0,j*ny_:(j+1)*ny_] = x[j]
        grid[1,j*ny_:(j+1)*ny_] = y

    return grid

def integrate(integrator,fixed_step_integrators,stride,atol,rtol,nx=1000,ny=500,x_min=0.,x_max=2.,y_min=0.,y_max=1.,t_start=0.,t_end=20.):
    if integrator.__name__ in fixed_step_integrators:
        advect_fixed(t_start,t_end,nx,ny,x_min,x_max,y_min,y_max,integrator,stride,doublegyre_wrapper)
    else:
        advect_adaptive(t_start,t_end,nx,ny,x_min,x_max,y_min,y_max,integrator,stride,doublegyre_wrapper,atol,rtol)


def advect_fixed(t_start,t_end,nx,ny,x_min,x_max,y_min,y_max,integrator,stride,rhs):
    grid_ = np.empty((5,2,(nx+4)*(ny+4)))
    dx = (x_max-x_min)/(nx-1)
    dy = (y_max-y_min)/(ny-1)

    dx_ = np.minimum(1.e-5,dx*1.e-2)
    dy_ = np.minimum(1.e-5,dy*1.e-2)

    grid_[0] = padded_grid_of_particles(nx,ny,x_min,x_max,y_min,y_max)
    grid_[1] = grid_[0] + np.array([dx_,0]).reshape(2,1)
    grid_[2] = grid_[0] + np.array([0,dy_]).reshape(2,1)
    grid_[3] = grid_[0] - np.array([dx_,0]).reshape(2,1)
    grid_[4] = grid_[0] - np.array([0,dy_]).reshape(2,1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    div = np.floor(np.linspace(0,(nx+4)*(ny+4),size+1)).astype(int)
    recv_buf = [np.empty((5,2,div[j+1]-div[j])) for j in range(size)]

    if rank == 0:
        grid_[:,:,div[rank]:div[rank+1]] = endpoints_fixed(t_start,t_end,grid_[:,:,div[rank]:div[rank+1]],stride,integrator,rhs)
        for j in range(1,size):
            comm.Recv(recv_buf[j],j)
            grid_[:,:,div[j]:div[j+1]] = recv_buf[j]
        ensure_path_exists('precomputed_advection/{}'.format(integrator.__name__))
        np.save('precomputed_advection/{}/nx={}_ny={}_t_start={}_t_end={}_stride={}_dx_m={}_dy_m={}_dx_a={}_dy_a={}.npy'.format(integrator.__name__,nx,ny,t_start,t_end,stride,dx,dy,dx_,dy_),grid_)

    else:
        comm.Send(endpoints_fixed(t_start,t_end,grid_[:,:,div[rank]:div[rank+1]],stride,integrator,rhs),dest=0)

def advect_adaptive(t_start,t_end,nx,ny,x_min,x_max,y_min,y_max,integrator,stride,rhs,atol,rtol):
    grid_ = np.empty((5,2,(nx+4)*(ny+4)))

    dx = (x_max-x_min)/(nx-1)
    dy = (y_max-y_min)/(ny-1)

    dx_ = np.minimum(1.e-5,dx*1.e-2)
    dy_ = np.minimum(1.e-5,dy*1.e-2)



    grid_[0] = padded_grid_of_particles(nx,ny,x_min,x_max,y_min,y_max)
    grid_[1] = grid_[0] + np.array([dx_,0]).reshape(2,1)
    grid_[2] = grid_[0] + np.array([0,dy_]).reshape(2,1)
    grid_[3] = grid_[0] - np.array([dx_,0]).reshape(2,1)
    grid_[4] = grid_[0] - np.array([0,dy_]).reshape(2,1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    div = np.floor(np.linspace(0,(nx+4)*(ny+4),size+1)).astype(int)

    recv_buf_grid = [np.empty((5,2,div[j+1]-div[j])) for j in range(size)]
    calls = [np.empty((5,div[j+1]-div[j]),dtype=int) for j in range(size)]
    if rank == 0:
        grid_[:,:,div[rank]:div[rank+1]],calls[:,div[j+1]-div[j]] = endpoints_adaptive(t_start,t_end,grid_[:,:,div[rank]:div[rank+1]],stride,integrator,rhs,atol,rtol)
        for j in range(1,size):
            comm.Recv(recv_buf_grid[j],j)
            comm.Recv(calls[j],j)
            grid_[:,:,div[j]:div[j+1]] = recv_buf_grid[j]
        ensure_path_exists('precomputed_advection/{}'.format(integrator.__name__))
        np.save('precomputed_advection/{}/nx={}_ny={}_t_start={}_t_end={}_stride={}_dx_m={}_dy_m={}_dx_a={}_dy_a={}.npy'.format(integrator.__name__,nx,ny,t_start,t_end,stride,dx,dy,dx_,dy_),grid_)
        mean_calls = np.empty(5)
        for j in range(5):
            mean_calls[j] = np.average(calls[j],weights=np.array([(div[i+1]-div[i])/((nx+4)*(ny+4)) for i in range(size)]))
        print('Mean calls:', mean_calls)
    else:
        ep,clls = endpoints_adaptive(t_start,t_end,grid_[:,:,div[rank]:div[rank+1]],stride,integrator,rhs,atol,rtol)
        comm.Send(ep,dest=0)
        comm.Send(clls,dest=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mt',help='Integrator name',type=str)
    parser.add_argument('st',help='(Initial) integration stride',type=np.float64)
    parser.add_argument('at',help='Absolute tolerance (adaptive methods)',type=np.float64)
    parser.add_argument('rt',help='Relative tolerance (adaptive methods)',type=np.float64)
    args = parser.parse_args()

    fixed_step_integrators = set([euler.__name__,rk2.__name__,rk3.__name__,rk4.__name__])
    integrators = [euler,rk2,rk3,rk4,rkbs32,rkbs54,rkdp54,rkdp87]
    for integ in integrators:
        if integ.__name__ == args.mt:
            integrator = integ
            break

    integrate(integrator,fixed_step_integrators,args.st,args.at,args.rt)
