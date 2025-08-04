import numpy as np
import math
# we will use a simple function y=3x+2
x=np.array([1,2,3,4,5])
y=np.array([5,8,11,14,17])
# initate the weigts
lr=0.1
iters=1000
m=len(x)
w=np.zeros(iters+1)
b=np.zeros(iters+1)
#initiate the adam hypoparameters
beta_1=0.9
beta_2=0.999
eps=10**-8          # to prevent dividing by zero
m_w=np.zeros(iters+1)
v_w=np.zeros(iters+1)
m_b=np.zeros(iters+1)
v_b=np.zeros(iters+1)
index=1
loss_prev=0
# calculate the loss 

for iter in range(1,iters+1):
    f_w_b=w[iter-1]*x+b[iter-1]
    loss=0
    dev_w=0
    dev_b=0
    gt_w=0
    gt_b=0      # gt is the dervitive
    for i in range(m):
        loss+=(y[i]-f_w_b[i])**2
        gt_w-=x[i]*(y[i]-f_w_b[i])
        gt_b-=(y[i]-f_w_b[i])


    loss/=2*m
    m_w[iter]=beta_1*m_w[iter-1]+(1-beta_1)*gt_w
    v_w[iter]=beta_2*v_w[iter-1]+(1-beta_2)*gt_w**2
    m_b[iter]=beta_1*m_b[iter-1]+(1-beta_1)*gt_b
    v_b[iter]=beta_2*v_b[iter-1]+(1-beta_2)*gt_b**2
    m_corr_w=m_w[iter]/(1-beta_1**iter)
    v_corr_w=v_w[iter]/(1-beta_2**iter)
    m_corr_b=m_b[iter]/(1-beta_1**iter)
    v_corr_b=v_b[iter]/(1-beta_2**iter)
    w[iter]=w[iter-1]-(lr*m_corr_w/(math.sqrt(v_corr_w)+eps))
    b[iter]=b[iter-1]-(lr*m_corr_b/(math.sqrt(v_corr_b)+eps))
    
    index+=1

print(index)
print(w[-1],b[-1])
    
 
    



