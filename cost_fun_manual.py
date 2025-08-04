import numpy as np


x=np.array(range(1,6))
y=np.array([5,8,11,14,17])
iters=1000
m=len(x) # m is the number of the samples
jw_b=np.empty(iters)  #jw_b cost
w=[0]
b=[0]
lr=0.001

for iter in range(iters):
    f_w_b=w[iter]*x+b[iter]
    jw_b[iter]=0
    dev_w=0
    dev_b=0
    for i in range(m):
        err=(y[i]-f_w_b[i])**2
        jw_b[iter]+=err
        dev_b+=y[i]-f_w_b[i]
        dev_w+=x[i]*(y[i]-f_w_b[i])
    jw_b[iter]/=m
    w.append(w[iter]+lr*2*dev_w/m)
    b.append(b[iter]+lr*2*dev_b/m)
    if abs(jw_b[iter]-jw_b[iter-1])==0:
        break
print(w[-1],b[-1],f_w_b)
