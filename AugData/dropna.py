import pandas as pd


def drop(originpath,nowpath):
    data = pd.read_csv(originpath)
    data = data.dropna(how="any")
    data.to_csv(nowpath)


drop("./needdrop/Similarword_origin.csv","./afterdrop/Similarword_origin.csv")
