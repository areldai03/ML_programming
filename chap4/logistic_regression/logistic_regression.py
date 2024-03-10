import numpy as np
import matplotlib.pyplot as plt

class Logistic:
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.dNum = x.shape[0]  # 学習データ数
        self.xDim = x.shape[1]  # 入力の次元数
        
        # 行列Xに「1」の要素を追加
        self.Z = np.concatenate([self.X,np.ones([self.dNum,1])],axis=1)
        self.a = 0.1
        self.w = np.random.normal(size=[self.xDim,1])
        self.b = np.random.normal(size=[1,1])
        self.small = 10e-8
    
    def update(self):
        p = self.predict(self.X)
        err = p - self.Y
        #logistic回帰の勾配の式
        grad = 1/self.dNum * np.matmul(self.Z.T, err)
        v = np.concatenate([self.w,self.b],axis=0)
        v -= self.a * grad
        self.w = v[:-1]
        self.b = v[[-1]]

    def predict(self, x):
        f = np.matmul(x, self.w) + self.b
        return 1/(1+np.exp(-f))
    
    def CE(self,X,Y):
        P= self.predict(X)
        return -np.mean(Y*np.log(P+self.small)+(1-Y)*np.log(1-P+self.small))
    
    def accuracy(self, X, Y):
        pred = self.predict(X)
        pred[pred<0.5] = 0
        pred[pred>0.5] = 1
        return np.mean(Y==pred)
    
    def plot_loss(self, trloss, teloss):
        plt.plot(trloss, 'r',label="train")
        plt.plot(teloss,'b',label="test")
        plt.show()

