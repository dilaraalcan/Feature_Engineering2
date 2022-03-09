import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler #standartlastırma dönüştürme.

pd.set_option('display.max_columns', None) #bütün sutunları göster
pd.set_option('display.max_rows', None) # bütün satırları göster
pd.set_option('display.float_format', lambda x: '%.3f' % x) #virgülden sonra 3 basamak göster.
pd.set_option('display.width', 500) #500 tane olsun satırlar ve sutunlar



import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC


def load():
    data = pd.read_csv("/Users/dlaraalcan/Desktop/DSMLBC/week6_Feature Engineering/datasets/titanic.csv")
    return data

df = load()
df.head()
df.shape

df.columns = [col.upper() for col in df.columns]
# değişken isimlerini düzenliyoruz. büyük harfe ceviriyoruz.
df.columns

# 1. Feature Engineering (Değişken Mühendisliği)
####################################################

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# cabin değişkeninde nan degerler var mı yok mu ?
# bool veri tipini (true,false) int'e cevir. 0-1


# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name değişkeninin harfleri toplamı

# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name degişkeninde kac tane kelime var?
# bosluklara göre split ederek kelime sayısını alır.

# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# dr ifadesiyle baslayan name degerleri


# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# name değişkeninde baslıkları düzenle
# boslukla baslayan sonrasında büyük kücük harf ve noktayla biten ifadeyi cıkar.

# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# akrabalık derecelerini ve kişinin kendisini topla
# ailedeki kişi sayısını bul

# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# yeni bir değişken olusturuluyor.
# kişi eger akraba sayısı sıfıra eşitse kişi yalnızdır.
# kişinin akraba sayısı sıfırdan büyükse kişi yalnız değildir.


# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# yas aralıgına göre isimlendirmeler.

# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'
# cinsiyete baglı yas aralıgı sınıflandırması


df.head()
df.columns


# değişken sınıflandırması

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # belirlenen degerler tamamen yoruma dayalı. farklı projede farklılık gösterebilir.
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

         mesela bilet sınıfı 1-2-3-4-5 olarak sayısal olarak gözükebilir ama kategoriktir.
         veya survived degişkeni hayatta kalma durumu 1-0 olarak ifade edilir numerik görülür fakat kategoriktir.


    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

                name kardinalliği yüksektir.
                ticket kardinalliği / çeşitliliği cok fazladır.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    # cat_cols, cat_but_car

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    # veri tipi object olanlar kategoriktir.

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    # cat_th= 10 olarak veri setinin durumuna göre belirlenmişti. yani 10 dan az ise bu numerik görünümlü kategorik olabilir yorumu yapılabilir.

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    # car_th= 20 olarak belirlenmişti.
    # eger ki kstegotik degişkende eşşiz deger sayısı 20'den fazlaysa bu kardinalliği yüksek demektir.
    # kategorik görünümlü kardinal deişkendir. mesela name degişkeni gibi.

    # daha sonra cat_cols listesini güncelleriz.
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]  # kardinalleri cıkar.

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]  # numerik gözüküp kategorik olanları cıkar.

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]
# passengerid numerik değişkenden cıkartılır.


# 2. Outliers (Aykırı Değerler)
#################################

# alt sınır üst sınır belirleme
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# aykırı deger var mı yok mu diye bakalım.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# numerik değişkenlerde aykırı degerlere bakalım
for col in num_cols:
    print(col, check_outlier(df, col))


# aykırı degerleri baskılama yöntemi: limitlere eşitleme
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)


# kontrol edelim
for col in num_cols:
    print(col, check_outlier(df, col))


# 3. Missing Values (Eksik Değerler)
######################################

df.isnull().values.any() # true: eksik deger var.

df.isnull().sum()
# eksik degerler:
#AGE                    177
#CABIN                  687
#NEW_AGE_PCLASS         177
#NEW_AGE_CAT            177
#NEW_SEX_CAT            177

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
# canin değişkeninde %77 oranında eksik veri var.
# cabin değişkeninde bulunanan eksik degerleri dolduramayız saglıklı olmaz bu nedenle sileriz.
df.drop("CABIN", inplace=True, axis=1)



remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

# age değişkenindeki eksik degerleri medyan ile doldur.
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

# yeni değişkenler:
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

# yas aralıgına baglı kategorileştirme:
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

#cinsiyet ve yas aralıgına baglı kategorileştirme:
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)


# 4. Label Encoding
#############################################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# iki parametreye sahip değişkenleri bulalım:
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

binary_cols
#['SEX', 'NEW_IS_ALONE']

# iki parametreye sahip kategorik değişkenleri numerik değişkene cevirelim:
for col in binary_cols:
    df = label_encoder(df, col)


# 5. Rare Encoding
#############################################

#Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SURVIVED", cat_cols)



def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

# 6. One-Hot Encoding
#############################################
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# df.drop(useless_cols, axis=1, inplace=True)


# 7. Standart Scaler
#############################################standartlastırma

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape

# Base Model
######################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

lgr = LogisticRegression()
lgr_model = lgr.fit(X_train, y_train)

# TRAIN ACCURACY
y_pred = lgr_model.predict(X_train)

# Accuracy
accuracy_score(y_train, y_pred)
# 0.8571428571428571


# TEST ACCURACY
y_pred = lgr_model.predict(X_test)
y_prob = lgr_model.predict_proba(X_test)[:, 1]

# Accuracy
accuracy_score(y_test, y_pred)
#  0.7798507462686567

# Precision
precision_score(y_test, y_pred)
# 0.7745098039215687


# Recall
recall_score(y_test, y_pred)
# 0.6869565217391305

# F1
f1_score(y_test, y_pred)
# 0.7281105990783411

print(classification_report(y_test, y_pred))
