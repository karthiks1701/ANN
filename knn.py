
import numpy as np

    

class knn:
    def  __init__(self,n=4,m=2):
        self.input=np.array([[1, 1, 1, -1],[-1, -1, -1, 1],[1, -1, -1, -1],[-1, -1, 1, 1]])
        self.n=n
        self.m=m
        self.W=np.array([[0.2,0.8],[0.6, 0.4],[0.5, 0.7],[0.9 ,0.3]])
        self.learningfactor=0.9
        self.decay=0.5
        self.noofepoch=1
        
        
    def euclidean_distance(self,inp):
        output=np.array([0,0])
        for i in range(0,self.m):
            print((self.W[:,i].T-inp)**2)
            output[i]=np.sum((self.W[:,i].T-inp)**2)
            print("output",output)
        return output
    
    def updation(self,j,inp1):
        #print(self.W[:,j])
        #print(self.learningfactor*(inp1.T-self.W[:,j]))
        self.W[:,j]=self.W[:,j]+self.learningfactor*(inp1.T-self.W[:,j])
        
        #print(self.W)
        
    def Training(self):
        for epoch in range(0,self.noofepoch):
            
            for inps in range(0,len(self.input)):
                print("weights",self.W)
                out=self.euclidean_distance(self.input[inps])
                x=np.argmin(out)
                self.updation(x,self.input[inps])
            self.learningfactor *= self.decay
    
    

if __name__ == '__main__':
    k=knn(4,2)
    k.Training()
    
    