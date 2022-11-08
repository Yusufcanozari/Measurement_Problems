import pandas as pd
import scipy.stats as st
import math
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

pd.set_option("display.max_columns",None)
pd.set_option("display.max_row",None)

df_ = pd.read_csv(r"Datasets/####.csv")#I can't share this data
df = df_.copy()
df.head()
df.dtypes

###############################################################################
###############################################################################
# GÖREV 1

df["overall"].mean() # current mean

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df["reviewTime"].max()
curren_Date = df["reviewTime"].max()

df["days"] = (curren_Date - df["reviewTime"]).dt.days

z = df["days"].quantile([0.25,0.5,0.75])

def day_diff(dataframe,w1=28,w2=26,w3=24,w4=22,z=df["days"].quantile([0.25,0.5,0.75])):
    return (dataframe.loc[dataframe["days"]<z.iloc[0] , "overall"].mean() * w1/100 + \
            dataframe.loc[(dataframe["days"]>= z.iloc[0]) & (dataframe["days"]<z.iloc[1]),"overall"].mean() * w2 /100 + \
            dataframe.loc[(dataframe["days"]>= z.iloc[1]) & (dataframe["days"]<z.iloc[2]),"overall"].mean() * w3/100 + \
            dataframe.loc[dataframe["days"]>=z.iloc[2] , "overall"].mean() * w4/100)

day_diff(df)


df.loc[df["days"]<280 , "overall"].mean() * 28/100
df.loc[(df["days"]>= 280) & (df["days"]<430),"overall"].mean()
df.loc[(df["days"]>= 430) & (df["days"]<600),"overall"].mean()
df.loc[df["days"]>=600 , "overall"].mean()

###############################################################################
###############################################################################
# GÖREV 2
df["helpful_no"] = df["total_vote"] - df["helpful_yes"] #ADIM 1


def score_pos_neg_diff(up,down):
    return (up-down)

df["score_pos_neg_diff"]=score_pos_neg_diff(up=df["helpful_yes"],down=df["helpful_no"])

def score_average_rating(up,down):
    if up + down == 0:
        return 0
    else:
        return up / (up+down)

df["score_average_rating"] = df.apply(lambda x:score_average_rating(up=x["helpful_yes"],down=x["helpful_no"]),axis=1)

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x:wilson_lower_bound(up=x["helpful_yes"],down=x["helpful_no"]),axis = 1)


df.sort_values(by="wilson_lower_bound",ascending=False).head(20)

