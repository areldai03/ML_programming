import numpy as np


#データ作成 [身長、体重]　10人分
data_h = np.random.normal(170, 5, 10)
data_w = np.random.normal(60, 5, 10)
dataset = [[h, w] for h, w in zip(data_h, data_w)]
dataset = np.array(dataset)

#分散と標準偏差
vars = np.var(dataset, axis=0)
stds = np.std(dataset, axis=0)

#分散共分散行列 bais y/n
cov_nobias = np.cov(dataset.T)
cov_bias = np.cov(dataset.T, bias=True)

#相関行列
corrcoef = np.corrcoef(dataset.T)

print(f'(データセット){dataset}')
print(f'(分散）身長：{vars[0]}, 体重:{vars[1]})')
print(f'(標準偏差）身長：{stds[0]}, 体重:{stds[1]})')
print(f'(分散共分散行列nobias：{cov_nobias})')
print(f'(分散共分散行列bias：{cov_bias})')
print(f'(相関行列）{corrcoef})')



