import numpy as np

class madaline:
    
    def __init__(self):
        self.input=np.array([[1, 1],[1, -1],[-1,1],[-1,-1]])
        self.inputbias=np.array([0.5,0.5])
        self.outputbias=-0.8
        self.weights=np.array([[0.1,0.2],[0.3,0.4]])
        self.outputweights=np.array([0.5,0.6])
        self.bias_weight=1;
        self.DesiredTarget=np.array([-1,1,1,-1])
        self.learningfactor=0.05
        self.maxepochs=50
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
    
    #def output(self,):
        
    
    
    def Training(self):
        for i in range(0,self.maxepochs):
            self.print_function()
            
            for j in range(0,self.noofoutputpatterns):
                
                sum=np.array([0,0])
                sum=self.inputbias+self.input[j].dot(self.weights)
                
                #print(self.input[j].dot(self.weights))
                
                self.hiddenoutput=self.activation(sum)
                print("\n")
                finalsum=self.outputbias+self.hiddenoutput.dot(self.outputweights)
                self.outactivation(finalsum,j)
                
                if(self.finaloutput[j]!=self.DesiredTarget[j]):
                    x=np.array([0,0])
                    if(self.finaloutput[j]==1):
                        k=-1
                        x[0]=sum[0]**2
                        x[1]=sum[1]**2
                        if x[0]<x[1]:
                            k=0
                        elif x[0]==x[1]:
                            k=3
                        else:
                            k=1
                        if(k!=2 and k!=3):   
                        
                            self.inputbias[k]=self.inputbias[k]+self.learningfactor*(1-sum[k])
                            transpose=self.weights.transpose()
                            transpose[k]=transpose[k]+self.learningfactor*(1-sum[k])*self.input[k]
                            self.weights=transpose.transpose()
                    
                        if(k==3):
                        
                            for t in range(0,2):
                                self.inputbias[t]=self.inputbias[t]+self.learningfactor*(1-sum[t])
                                transpose=self.weights.transpose()
                                transpose[t]=transpose[t]+self.learningfactor*(1-sum[t])*self.input[t]
                                self.weights=transpose.transpose()
                    
                    elif(self.finaloutput[j]==-1):
                        r=-1
                        x=sum
                        if (x[0]>0 and x[1]>0) :
                            r=3
                        elif (x[0]>0 and x[1]<0):
                            r=0
                        elif (x[1]>0 and x[0]<0):
                            r=1
                        elif (x[1]<0 and x[0]<0):  
                            r=2
                             
                    
                    
                        if(r!=2 and r!=3 ):   
                         
                            self.inputbias[r]=self.inputbias[r]+self.learningfactor*(-1-sum[r])
                            transpose=self.weights.transpose()
                            transpose[r]=transpose[r]+self.learningfactor*(-1-sum[r])*self.input[r]
                            self.weights=transpose.transpose()
                    
                        if(r==3):
                        
                            for t in range(0,2):
                                self.inputbias[t]=self.inputbias[t]+self.learningfactor*(-1-sum[t])
                                transpose=self.weights.transpose()
                                transpose[t]=transpose[t]+self.learningfactor*(-1-sum[t])*self.input[t]
                                self.weights=transpose.transpose()
                            


if __name__ == "__main__":
    m=madaline()
    m.Training()