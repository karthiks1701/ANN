import numpy as np

class madaline:
    
    def __init__(self):
        self.input=np.array([[1, 1], [-1,1], [1, -1],[-1,-1]])
        self.inputbias=np.array([0.5,0.5])
        self.outputbias=0.5
        self.weights=np.array([[0.1,0.2],[0.3,0.4]])
        self.outputweights=np.array([0.5,0.6])
        self.bias_weight=1;
        self.DesiredTarget=np.array([1,-1,1,1])
        self.learningfactor=0.7
        self.maxepochs=4
        self.noofinputs=2
        self.noofoutput=1
        self.noofoutputpatterns=4
        self.hiddenoutput=np.array([0,0])
        self.finaloutput=np.array([0,0,0,0])
        
        
    def print_function(self):
        print("weights      ",self.weights)
        print("bias_weights ",self.inputbias)
        print("final_output ",self.finaloutput)
        print("\n")
    
    def outactivation(self,sum,j):
        if sum>=0:
            self.finaloutput[j]=1
        elif sum<0:
            self.finaloutput[j]=-1
    
    def activation(self,sum):
        for i in range(0,2):
            if sum[i]>=0:
                sum[i]=1
            elif sum[i]<0:
                sum[i]=-1
        return sum

        
    
    
    def Training(self):
        for i in range(0,self.maxepochs):
            self.print_function()
            
            for j in range(0,self.noofoutputpatterns):
                
                sum=np.array([0,0])
                sum=self.inputbias+(self.input[j].T).dot(self.weights)
                print(sum)
              
                
                self.hiddenoutput=self.activation(sum.copy())
                print(self.hiddenoutput)
                finalsum=self.outputbias+(self.hiddenoutput.T).dot(self.outputweights)
                print(finalsum)
                self.outactivation(finalsum,j)
                
                
                if(self.finaloutput[j]!=self.DesiredTarget[j]):
                    if(self.DesiredTarget[j]==1):
                        k = (sum**2).argmin()
                        print(str(1), k)
                        self.inputbias[k]=self.inputbias[k]+self.learningfactor*(1-sum[k])
                        self.weights[:,k]=self.weights[:,k]+self.learningfactor*(1-sum[k])*self.input[j].T
                    
                    elif(self.DesiredTarget[j]==-1):
                        
                        print("r", sum)
                        for r in range(0,len(sum)):
                            if sum[r] > 0:
                                print(str(-1), r)
                                self.inputbias[r]=self.inputbias[r]+self.learningfactor*(-1-sum[r])
                                self.weights[:,r]=self.weights[:,r]+self.learningfactor*(-1-sum[r])*self.input[j].T
                                
                                


if __name__ == "__main__":
    m=madaline()
    m.Training()
