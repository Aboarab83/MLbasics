import numpy as np


x=np.array(range(1,6))
y=np.array([5,8,11,14,17])
iters=1000
m=len(x) # m is the number of the samples
jw_b=np.empty(iters+1)  #jw_b cost
w=np.zeros(iters+1)
b=np.zeros(iters+1)
lr=0.01

for iter in range(1,iters+1):
    f_w_b=w[iter-1]*x+b[iter-1]
    jw_b[iter]=0
    dev_w=0
    dev_b=0
    for i in range(m):
        err=(y[i]-f_w_b[i])**2
        jw_b[iter]+=err
        dev_b+=y[i]-f_w_b[i]
        dev_w+=x[i]*(y[i]-f_w_b[i])
    jw_b[iter]/=m
    w[iter]=w[iter-1]+lr*2*dev_w/m
    b[iter]=b[iter-1]+lr*2*dev_b/m

print(w[-1],b[-1],f_w_b)
