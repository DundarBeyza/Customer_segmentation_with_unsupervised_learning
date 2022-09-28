# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv.csv verisini okuyunuz.
           # 2. Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
           # Tenure(Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi
           # yeni değişkenler oluşturabilirsiniz.


#!pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import datetime as dt
from scipy import stats


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)
import warnings
warnings.filterwarnings("ignore")



# GÖREV 1 : Veri Setini Okutma

df_ = pd.read_csv("flo_unsupervised_learning/flo_data_20k.csv")
df = df_.copy()

# helpers dosyası oluşturup sık kullanılan fonksiyonlarınızı çağırın
from helpers.eda import *
from helpers.data_prep import *


check_df(df)


# Tarih değişkenlerini çevirme

df.columns[df.columns.str.contains("date")]

date_columns = df.columns[df.columns.str.contains("date")]

df[date_columns].apply(pd.to_datetime)

df[date_columns] = df[date_columns].apply(pd.to_datetime)

df.info()

# Müşterinin son alışveriş tarihi

df["last_order_date"].max()

analysis_date = dt.datetime(2021, 6, 1)

# Değişken Oluşturma

(analysis_date - df["last_order_date"]).astype("timedelta64[D]")

df["recency"] = (analysis_date - df["last_order_date"]).astype("timedelta64[D]")

df["tenure"] = (df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")

df.head()

model_df = df[["order_num_total_ever_online", "order_num_total_ever_offline" ,"customer_value_total_ever_offline",
               "customer_value_total_ever_online", "recency", "tenure"]]


## GÖREV 2 : K - Means ile Müşteri Segmentasyonu


# 1. Değişkenleri standartlaştırınız.
#SKEWNESS
#Burada ilk başta değişkenlerimizin çarpıklığına bakarız.


def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1);
check_skew(model_df, 'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df, 'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df, 'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df, 'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df, 'recency')
plt.subplot(6, 1, 6)
check_skew(model_df, 'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show()


model_df_transform_list = ["order_num_total_ever_online",
                           "order_num_total_ever_offline" ,
                           "customer_value_total_ever_offline",
                           "customer_value_total_ever_online",
                           "recency",
                           "tenure"]

for col in model_df_transform_list:
    model_df[col] = np.log1p(model_df[col])



sc = MinMaxScaler((0, 1))
model_sc = sc.fit_transform(model_df)

model_df = pd.DataFrame(model_sc, columns=model_df.columns)

model_df.head()

## 2 : Optimum Küme Sayısının Belirlenmesi

## K-Means sadece sayısal değişkenlerle çalışır !!!! ##

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()

## 3 - Model Oluşturma

kmeans = KMeans(n_clusters=7 , random_state=42).fit(model_df)

segments = kmeans.labels_

pd.DataFrame(segments).value_counts()

df.info()

df.head()

final_df = df[["master_id",
               "order_num_total_ever_online",
                "order_num_total_ever_offline",
                "customer_value_total_ever_offline",
                "customer_value_total_ever_online",
                "recency",
                "tenure"]]

final_df["segments"] = segments

final_df.head()


final_df.groupby("segments").agg({"order_num_total_ever_online":["median","min","max"],
                                  "order_num_total_ever_offline":["median","min","max"],
                                  "customer_value_total_ever_offline":["median","min","max"],
                                  "customer_value_total_ever_online":["median","min","max"],
                                  "recency":["median", "min", "max"],
                                  "tenure":["median", "min", "max"]})


## GÖREV 3 : Hierarchical Clustering ile Müşteri Segmentasyonu


hc_complete = linkage(model_df, "ward")

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=5,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=10, color='r', linestyle='--')
plt.show()


from sklearn.cluster import AgglomerativeClustering


hc = AgglomerativeClustering(n_clusters=5)   #birleştirici
segments = hc.fit_predict(model_df)
pd.DataFrame(segments).value_counts()

final_df = df[["master_id",
               "order_num_total_ever_online",
                "order_num_total_ever_offline",
                "customer_value_total_ever_offline",
                "customer_value_total_ever_online",
                "recency",
                "tenure"]]

final_df["segments"] = segments

final_df.head()

final_df.groupby("segments").agg({"order_num_total_ever_online":["median","min","max"],
                                  "order_num_total_ever_offline":["median","min","max"],
                                  "customer_value_total_ever_offline":["median","min","max"],
                                  "customer_value_total_ever_online":["median","min","max"],
                                  "recency":["median", "min", "max"],
                                  "tenure":["median", "min", "max"]})





