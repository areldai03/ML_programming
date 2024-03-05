import pandas as pd
import numpy as np

class Data:
    
    def __init__(self, path):
        self.data = pd.read_csv(path)
    
    def preprocess(self):
        #説明変数を居住面積を抽出
        #単回帰分析
        self.x = self.data[self.data['MSSubClass']==60][['GrLivArea']].values
        self.y = self.data[self.data['MSSubClass']==60][["SalePrice"]].values

