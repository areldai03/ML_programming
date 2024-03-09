import data_make as dm
import logistic_regression as lr

def main():
    #make data
    data = dm.Data(r"./chap4/linear_regression\train.csv")
    data.preprocess()
    split_num = int(len(data.x) * 0.9)
    xte = data.x[split_num:]
    yte = data.y[split_num:]
    

    #train
    logistic = lr.Logistic(data.x,data.y)
    for i in range(1000):
        if i % 100 == 0:
            print(f"{i}: 損失{logistic.CE(xte,yte)}, パラメタ：{logistic.w},{logistic.b}, 正解率：{logistic.accuracy(xte,yte)}")
        
        logistic.update()

if __name__=="__main__":
    main()

