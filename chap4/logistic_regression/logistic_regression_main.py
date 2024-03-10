import data_make as dm
import logistic_regression as lr

def main():
    #make data
    data = dm.Data(r"./chap4/linear_regression\train.csv")
    data.preprocess()
    split_num = int(len(data.x) * 0.9) 
    xte = data.x[split_num:]
    yte = data.y[split_num:]
    xtr = data.x[:split_num]
    ytr = data.y[:split_num]


    #train
    tr_loss = []
    te_loss = []

    logistic = lr.Logistic(xtr, ytr)
    for i in range(1000):
        tr_loss.append(logistic.CE(xtr,ytr))
        te_loss.append(logistic.CE(xte,yte))
        if i % 100 == 0:
            print(f"{i}: 損失{logistic.CE(xte,yte)}, パラメタ：{logistic.w},{logistic.b}, 正解率：{logistic.accuracy(xte,yte)}")
        
        logistic.update()
    
    #分析
    logistic.plot_loss(tr_loss, te_loss)

if __name__=="__main__":
    main()

