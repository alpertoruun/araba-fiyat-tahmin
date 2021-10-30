import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import r2_score
data=pd.read_csv("C:/Users/Alper Torun/Desktop/kişisel gelişim/yazılım/Python/proje geliştirme/3-araba fiyat tahmin/Automobile_data.csv")

data.shape
data.info()
#veri eksik deger düzenlemesi
data=data.replace(-1,np.NaN)
data=data.replace("?",np.NaN)
#sayısal objectleri, sayiya cevirme
for column in ['normalized-losses',"bore","stroke","horsepower","peak-rpm","price"]:
    data[column]=data[column].astype(np.float)
data.isnull().sum()
#eksik degerlerin doldurulmasi
for column in data.columns:
    if data[column].dtypes != "object" and data.isnull().sum()[column]>0:
        data[column]=data[column].fillna(data[column].mean()) #tum numeric sutunlari doldurma
data.isnull().sum()

#kategorilere bakma
{column: list(data[column].unique()) for column in data.columns if data.dtypes[column] == 'object'}

numeric_ordering ={
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'eight': 8,
    'twelve': 12
}

data['num-of-cylinders'] = data['num-of-cylinders'].replace(numeric_ordering)
data['num-of-doors'] = data['num-of-doors'].apply(lambda x: 0 if x == 'two' else 1)
print("Total missing values:", data.isna().sum().sum())

def binary_encode(df, columns, positive_values):
    df = df.copy()
    for column, positive_value in zip(columns, positive_values):
        df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
    return df


binary_features = [
    'fuel-type',
    'aspiration',
    'engine-location',
]

binary_positive_values = [
    'diesel',
    'turbo',
    'front'
]

data = binary_encode(
    data,
    columns=binary_features,
    positive_values=binary_positive_values
)

def onehot_encode(df, columns, prefixes):
    df = df.copy()
    for column, prefix in zip(columns, prefixes):
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df

nominal_features = [
    'make',
    'body-style',
    'drive-wheels',
    'engine-type',
    'fuel-system'
]

prefixes = [
    'MK',
    'BS',
    'DW',
    'ET',
    'FS'
]

data = onehot_encode(
    data,
    columns=nominal_features,
    prefixes=prefixes
)
y = data['price'].copy()
X = data.drop('price', axis=1).copy()
scaler = StandardScaler()

X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=20)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=20)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {'learning_rate': 0.001, 'max_depth': 6}

model = xgb.train(params, dtrain, evals=[(dval, 'eval')], num_boost_round=10000, early_stopping_rounds=10, verbose_eval=False)

y_true = np.array(y_test, dtype=np.float)
y_pred = np.array(model.predict(dtest), dtype=np.float)
print("R^2 Score: {:.4f}".format(r2_score(y_true, y_pred)))

