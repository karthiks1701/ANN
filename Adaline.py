import numpy as np

class adaline:
    
    
    def __init__(self):
        self.input=np.array([[1, 1],[1, -1],[-1,1],[-1,-1]])
        self.bias=1
        self.weights=np.array([1,-1])
        self.bias_weight=1;
        self.DesiredTarget=np.array([1,1,1,-1])
        self.toll=0.4
        self.learningfactor=0.05
        self.maxepochs=50
        self.noofinputs=2
        self.noofoutput=1
        self.output1=0;
        self.noofoutputpatterns=4
        self.finaloutput=np.array([0,0,0,0])
        self.flag=0
    
    def print_function(self):
        print("weights      ",self.weights)
        print("bias_weights ",self.bias_weight)
        print("final_output ",self.finaloutput)
        print("\n") 
    
    def activation(self,sum,j):
        if sum>=0:
            self.finaloutput[j]=+1
        elif sum<0:
            self.finaloutput[j]=-1;
        
    def output(self,j):
        return self.input[j].dot(self.weights)+self.bias*self.bias_weight
    
    def weight_updation(self,sum,j):
        
        x=(self.DesiredTarget[j]-sum)        
        self.weights=self.weights+self.learningfactor*x*(self.input[j])
        self.bias_weight=self.bias_weight+self.learningfactor*x
        
    
       
    def Training(self):
        
        for i in range(0,self.maxepochs):
            
            self.print_function()
            
            for j in range(0,self.noofoutputpatterns):
                sum=0
                
                sum=self.output(j)
                
                self.activation(sum,j)
                
                prevweights=self.weights
                prevbias_weight=self.bias_weight
               
                self.weight_updation(sum,j)
                
                deltachange=self.weights-prevweights
                c=np.amax(deltachange)
                if c > self.toll:
                    self.flag=1
                    break
            
     
            
            if self.flag==1:
                break
                


if __name__ == "__main__":
    
    a=adaline() 
    a.Training()
    
    
    
    pass

































































