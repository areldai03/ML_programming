import numpy as np
import matplotlib.pylab as plt

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dNum = x.shape[0]  # 学習データ数
        self.xDim = x.shape[1]  # 入力の次元数
    

    # 最小二乗法を用いてモデルパラメータを最適化
    def train(self):
        Z = np.concatenate([self.x,np.ones([self.dNum,1])],axis=1)
        ZZ = 1/self.dNum * np.matmul(Z.T,Z)
        ZY = 1/self.dNum * np.matmul(Z.T,self.y)
        # パラメータv
        v = np.matmul(np.linalg.inv(ZZ), ZY)

        self.w = v[:-1]
        self.b = v[-1]

    def predict(self, x):
        return np.matmul(x, self.w) + self.b
    

    def RMSE(self, x, y):
        return np.sqrt(np.mean(np.square(self.predict(x)-y)))
    
    def R2(self, x, y):
        return 1 - np.sum(np.square(self.predict(x)-y))/np.sum(np.square(y-np.mean(y, axis=0)))
    

    def plot(self, x=[], y=[]):
        if x.shape[1] != 1:
            return
        
        fig = plt.figure(figsize=(8,5),dpi=100)
        
        # 線形モデルの直線の端点の座標を計算
        Xlin = np.array([[0],[np.max(x)]])
        Yplin = self.predict(Xlin)

        # データと線形モデルのプロット
        plt.plot(x,y,'.',label="データ")
        plt.plot(Xlin,Yplin,'r',label="線形モデル")
        plt.legend()
        
        # 各軸の範囲とラベルの設定
        plt.ylim([0,np.max(y)])
        plt.xlim([0,np.max(x)])
        plt.show()


