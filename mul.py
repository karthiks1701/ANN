import numpy as np
    
 
a=np.random.random_integers(1,3,size=[2,2])
b=np.random.random_integers(1,3,size=[2,1])
print(a)
print(b)
print(np.dot(b.T,a))