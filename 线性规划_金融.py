import numpy as np;
import matplotlib.pyplot as plt
import scipy.optimize as op
a=0
profit_list=[]
a_list=[]
while a<0.05:
    x1=(0,None)
    c=np.array([-0.05,-0.27,-0.19,-0.185,-0.185])
    A=np.hstack((np.zeros((4,1)),np.diag([0.025,0.015,0.055,0.026])))
    b=a*np.ones((4,1))
    Aeq=np.array([[1,1.01,1.02,1.045,1.065]])
    beq=np.array([1])
    res=op.linprog(c,A,b,Aeq,beq,bounds=(x1,x1,x1,x1,x1))
    profit=-res.fun
    profit_list.append(profit)
    a_list.append(a)
    a+=0.001

plt.figure(figsize=(10,7))
plt.plot(a_list,profit_list)
plt.xlabel('a');plt.ylabel('profit')
plt.show()
