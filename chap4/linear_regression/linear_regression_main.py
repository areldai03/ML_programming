import data_make as dm
import linear_regression as lr

def main():
    # データ準備
    data = dm.Data(r".\chap4\linear_regression\train.csv")
    data.preprocess()
    split_num = int(len(data.x) * 0.9)
    Xtr = data.x[:split_num]
    Ytr = data.y[:split_num]
    Xte = data.x[split_num:]
    Yte = data.y[split_num:]

    # モデルの学習
    model = lr.LinearRegression(Xtr, Ytr)
    model.train()

    # 評価
    print(f'パラメータ）w:{model.w}, b:{model.b}')
    print(f'平方平均二乗誤差:{model.RMSE(Xte, Yte):.2f}')
    print(f'決定係数:{model.R2(Xte, Yte)}')

    model.plot(x=Xtr, y=Ytr)

if __name__ == "__main__":
    main()
