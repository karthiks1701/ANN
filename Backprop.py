import numpy as np

class neuralnet():
    def __init__(self,output,inp,hidden):
        self.wi=np.random.randn(hidden,inp)
        self.wh=np.random.randn(output,hidden)
        self.bi=np.random.randn(hidden,1)
        self.bo=np.random.randn(output,1)
        self.learningrate=0.5
        self.epochs=500
    
    def sigmoid(self,inp,der=True):
        if der==True:
            return 1/(1+np.exp(-1*inp))
        else:
            return np.exp(-1*inp)/((1+np.exp(-1*inp))**2)
        
    def forward(self,inp):
        self.zi = np.dot(self.wi, inp) + self.bi
        self.a = self.sigmoid(self.zi)
        self.zh = np.dot(self.wh, self.a) + self.bo
        print(self.sigmoid(self.zh))
        return self.sigmoid(self.zh)
    
    def loss(self,x,y):
        return (np.sum(x - y)**2)
    
    def Train(self,data):
        for epoch in range(self.epochs):
            dWh, dbh, dWi, dbi, m, l = np.zeros(self.wh.shape), np.zeros(self.bo.shape), np.zeros(self.wi.shape), np.zeros(self.bi.shape), len(data), 0
            for (x, y) in data:
                y_hat = self.forward(x)
                l += self.loss(y_hat, y)
                dzh = 2*(y_hat - y)*self.sigmoid(self.zh,False)
                dWh += dzh*self.a.T
                dbh += dzh
                da = np.sum(dzh*self.wh, axis=0).reshape((-1, 1))
                dzi = self.sigmoid(self.zi,False)
                dWi += dzi*x.T
                dbi += dzi
            l /= m
            dWh /= m
            dbh /= m
            dWi /= m
            dbi /= m
            self.wi -= self.learningrate*dWi
            self.bi -= self.learningrate*dbi
            self.wh -= self.learningrate*dWh
            self.bo -= self.learningrate*dbh
        print(self.wi)
        print(self.wh)
        print(self.bi)
        print(self.bo)
        
        
        
if __name__ == '__main__':
        NN=neuralnet(1,2,2)
        NN.Train([(np.array([[1],[-1]]),np.array([[1]])),(np.array([[-1],[1]]),np.array([[1]])),(np.array([[1],[1]]),np.array([[-1]])),(np.array([[-1],[-1]]),np.array([[-1]]))])
        NN.forward([(np.array([[-1],[1]]))])
      
        
                 