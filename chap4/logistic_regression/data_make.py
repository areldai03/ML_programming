import pandas as pd
import numpy as np

class Data:
    
    def __init__(self, path):
        self.data = pd.read_csv(path)
    
    def preprocess(self):
        self.x = self.data[(self.data['MSSubClass']==30) |(self.data['MSSubClass']==60)][['GrLivArea']].values
        self.y = self.data[(self.data['MSSubClass']==30) |(self.data['MSSubClass']==60)][['MSSubClass']].values
        self.y[self.y==30] = 0
        self.y[self.y==60] = 1
        mean_x = np.mean(self.x)
        std_x = np.std(self.x)
        self.x = (self.x-mean_x)/std_x

