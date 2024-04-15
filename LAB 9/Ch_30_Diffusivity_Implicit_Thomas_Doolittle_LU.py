import xlwings as xw
import numpy as np

@xw.sub
def diffsivity_implicit_py():
# obtain workbook/worksheet objects
    wb = xw.Book.caller() # workbook from which this executable @xw.sub was called
    data_sht = wb.sheets['Data'] # read data from this sheet
    results_sht = wb.sheets['Results'] # output results to this sheet
# read data
    xL = data_sht.range(1,1).value # x-coord of left boundary [L]
    xR = data_sht.range(2,1).value # x-coord of right boundary [L]
    BCL_type = data_sht.range(3,1).value # true means T specified as left BC, false means dT/dx specified as left BC
    BCL_val = data_sht.range(4,1).value # value of specified T [T], or specified dT/dx [T/L] at left BC
    BCR_type = data_sht.range(5,1).value # true means T specified as left BC, false means dT/dx specified as left BC
    BCR_val = data_sht.range(6,1).value # value of specified T [T], or specified dT/dx [T/L] at left BC
    T_init = data_sht.range(7,1).value # initial temperature [T] on domain, t=0, xL <= x <= xR
    k = data_sht.range(8,1).value # thermal diffusivity [L^2/t]
    N = int(data_sht.range(9,1).value) # delta_x = (xR-xL)/N, where delta-x in units of [L]
    delta_t = data_sht.range(10,1).value # delta_t, time step length [t]
    max_t = data_sht.range(11,1).value # time to end simulation [t]
#  set up arrays (elements subscripts 0 to N) and calculate variables
    T_beg = np.zeros(N+1) # temperature at beginning of step [T]
    T_end = np.zeros(N+1) # temperature at end of step [T]
    e = np.zeros(N+1) # coefficient of tridiagonal system of equations, left of main diagonal
    f = np.zeros(N+1) # coefficient of tridiagonal system of equations, main diagonal
    g = np.zeros(N+1) # coefficient of tridiagonal system of equations, right of main diagonal
    r = np.zeros(N+1) # coefficient of tridiagonal system of equations, right hand side vector
    delta_x = (xR-xL)/N
    lambda_ = k*delta_t/delta_x**2 # Using "lambda_" since  "lambda" is reserved in python (shows in blue font)
# initialize results worksheet
    results_sht.clear_contents()
    results_sht.range(1,2).value = 'x -->'
    r = np.linspace(xL,xR,num=N+1) # temporary usage of array r to store values of x coord. for output
    results_sht.range((1,3),(1,N+3)).value = r # temporary usage of array r to output values of x
    results_sht.range(2,1).value = 't'
# initial temperature
    T_beg[0:N+1] = T_init
# matrix coefficients (matrix does not change over time), internal points 1 to N-1
    e[1:N] = -lambda_
    f[1:N] = 1. + 2*lambda_
    g[1:N] = -lambda_
# left BC, 1st equation
    if BCL_type:
        f[0] = 1.
        g[0] = 0.
    else:
        f[0] = 1. + 2*lambda_
        g[0] = -2*lambda_
# right BC, last equation
    if BCR_type:
        e[N] = 0.
        f[N] = 1.
    else:
        e[N] = -2*lambda_
        f[N] = 1. + 2*lambda_
# set values before 1st time step
    t = 0.
    irow = 3
    thomas_decomp(e,f,g,N) # decompose matrix only once (because matrix doesn't change with time)
# begin time step loop
    while True: # this while gives an "endless loop", see break conditional at end
        t += delta_t
# left BC
        if BCL_type:
            r[0] = BCL_val # Dirichlet BC
        else:
            r[0] = T_beg[0] - 2*lambda_*delta_x*BCL_val # Neumann BC
# right BC
        if BCR_type:
            r[N] = BCR_val # Dirichlet BC
        else:
            r[N] = T_beg[N] + 2*lambda_*delta_x*BCR_val # Neumann BC
# right hand side of equations ( 1 to N-1 )
        r[1:N] = T_beg[1:N]
# solve matrix for T_end
        thomas_solve(e,f,g,r,T_end,N) # uses values of e and f, after thomas_decomp
# output results
        results_sht.range(irow,1).value = t
        results_sht.range(irow,2).value = 'T(x,t) -->'
        results_sht.range((irow,3),(irow,3+N)).value = T_end
# prepare for next step
        irow += 1
        T_beg = np.copy(T_end) # end of last step is beginning of next step
 # epsilon=1.E-6 so "within epsilon" prevents taking an extra step in case
 # roundoff causes t to land just slightly less than max_t
        if t > (max_t - 1.E-6):
            break # finished time steps, end execution


def thomas_decomp(e,f,g,N):
    '''Based on pseudocode in Fig. 11.2, Step (a), the Thomas Algorithm is
    Doolittle LU decomposition of an (N+1)x(N+1) tridiagonal matrix 
    (see Section 10.1.1 and Fig. 11.2)

    Calculates modified e and f vectors to be used with thomas_solve()'''
    for k in range(1,N+1):
        e[k] /= f[k-1]
        f[k] -= e[k]*g[k-1]

def thomas_solve(e,f,g,r,x,N):
    ''' Based on pseducode in Fig 11.2, Steps (b) and (c), the Thomas Algorithm
    is Doolittle LU decomposition of an (N+1)x(N+1) tridiagonal matrix 
    (see Section 10.1.1 and Fig. 11.2)

    Given modified e and f vectors from thomas_decompose(), do forward substitution
    (modifying the r vector), followed by backward substitution to calculate
    the solution vector, x'''
    for k in range(1,N+1):
        r[k] -= e[k]*r[k-1]
    x[N] = r[N] / f[N]
    for k in range(N-1,-1,-1):
        x[k] = (r[k] - g[k]*x[k+1]) / f[k]

if __name__ == '__main__':
    xw.serve()