import itertools
import time
from numbers import Real
from random import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_inline
import xgboost as xg

import tensorflow as tf
import seaborn as sns #heatmap haritaları için kullanılıyor.
from datetime import datetime
from IPython.display import set_matplotlib_formats
from keras import Input
from keras.callbacks import EarlyStopping
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA



matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Softmax, Conv2D, MaxPooling2D, Flatten, MaxPool2D, Conv1D, MaxPooling1D
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Masking
from keras.optimizers import  Adam
from keras_tuner.tuners import BayesianOptimization
from skopt import BayesSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
np.random.seed(42)
tf.random.set_seed(42)
# 1. VERİLERİ ÖN VE ANA AŞAMA OLARAK AYIRMA (SAN ANDREAS)

"""
filePath = "datasets/SA-1923-2023/" + "veriler_ayrilmamis(ana+on_asama)" + ".csv"
dfEqModel = pd.read_csv(filePath)

dfEqModel["timestamps"] = dfEqModel["time"]
dfEqModel.set_index('time', inplace=True)

#train_set, test_set = train_test_split(dfEqModel, test_size=0.33, random_state=None)# ön tahmin aşaması için veri setini %80 %20 böl
#ve bu %20'lik kısmı svm ile ön tahmin aşamasında hem eğitim hem test için kullan

test_rate = 0.1

# Veriyi bölmek için indeksleri hesapla
#split_index = int(len(dfEqModel) * test_rate)

# Train ve test verilerini ayır
pre_set = dfEqModel[:999]
main_set = dfEqModel[999:]

pre_set.to_csv('datasets/SA-1923-2023/ontahminasamasi_veri_seti.csv', index=False)# ön tahmin için kullanılacak verileri farklı bir csv dosyasına kaydet
main_set.to_csv('datasets/SA-1923-2023/anaasama_veri_seti.csv', index=False)# ön tahmin için kullanılacak verileri farklı bir csv dosyasına kaydet
"""

# 2. ÖN AŞAMA TAHMİN (SVM)# %25 hata oranı ile ağırlıklı ortlama seçildi

"""
peFilePath = 'datasets/SA-1923-2023/ontahminasamasi_veri_seti.csv'
prestimation = pd.read_csv(peFilePath)
prestimation = prestimation.drop(["time"], axis=1)
magM = prestimation[["mag_max"]] #y

y = magM
# SCALING
# sc = RobustScaler() # aykırı değerlere karşı dayanıklı
sc = MinMaxScaler() # en başarılı mae ve mse değerleri bu scale türünde çıktı mae:0.12, mse:0.02
x = prestimation.drop(["mag_max"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, shuffle=False)

print("x_train", x_train)
print("x_test", x_test)
print("y_train", y_train)
print("y_test", y_test)

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)


# GRIDSEARCHCV İLE EN İYİ PARAMETRE DEĞERLERİNİ BULMA

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': [0.1, 1, 10]
}

grid_search = GridSearchCV(estimator=svm.SVR(), param_grid=param_grid)
grid_search.fit(x_train, y_train)
print("En iyi parametreler:", grid_search.best_params_) #En iyi parametreler: {'C': 10, 'gamma': 1, 'kernel': 'rbf'}

svr = svm.SVR(kernel='rbf', C= 10, gamma= 0.1) # GridSearchCV ile optimum parametreler bulundu. MSE 0.009151399941817176, MAE 0.08548841651941114

svr.fit(x_train, y_train)
y_pred_svr = svr.predict(x_test) # y_pred


print("korelasyon", prestimation.corr())
pree=prestimation.corr()
filePath = "datasets/SA-1923-2023/" + "correlation" + ".csv"
pree.to_csv(filePath)
"""

# 3. EŞİK DEĞERİ BULMA

"""
# 3.1 BASİT ORTALAMA HESAPLAMASI %48.144074829637844 Hata oranı

percentage_residuals = []
residuals = y_test - y_pred_svr
absolute_residuals = np.abs(residuals) #mutlak değer ile farkını al

for index in range(len(y_test)):
    if y_test[index] != 0:
      percentage_residuals = (absolute_residuals[index] / y_test[index]) * 100
threshold = np.sum(percentage_residuals) / len(y_test)
print("thresholdbasitort", threshold )


# 3.2 AĞIRLIKLI ORTALAMA HESAPLAMASI #2.5878293439475556

weight = x_test[: , 13] #Weight(Mw) kolonunu al
print(len(weight))
print(weight)
errors  = y_test - y_pred_svr

weightedAverage = np.sum(np.abs(errors) * weight) / np.sum(weight) * 100

weightedAverage = weightedAverage / len(weight) # len(errors)'da kullanılabilir zaten iki dizinin boyutu aynı (741)
print("thresholdagirlik" , weightedAverage) #threshold değeri

# 3.3 STANDART SAPMA İLE EŞİK DEĞERİ BULMA  %136.24366457417756 Hata oranı
errors = y_test - y_pred_svr
standart_sapma_hatalar = np.std(errors)

# Gerçek değerlerin standart sapmasını hesaplayın
standart_sapma_gercek = np.std(y_test)

# Hata yüzdesini hesaplayın
hata_yuzdesi = (standart_sapma_hatalar / standart_sapma_gercek) * 100
print("thresholdstsapma", hata_yuzdesi)


# SVR MODELİN BAŞARISINI DEĞERLENDİRME
mse = mean_squared_error(y_test, y_pred_svr)
mae = mean_absolute_error(y_test, y_pred_svr)
print("MSE", mse)
print("MAE", mae)
"""

# 4. ANA AŞAMA TAHMİN

# 4.1. EŞİK DEĞERİNE (HATA ORANI) GÖRE VERİLERİ AYIRMA
"""
mainStage1=[] #Eşik Değerinden Büyük
mainStage2=[] #Eşik Değerinden Küçük

weightedAverage=[]
errors= []
weight= []

filePathSA = 'datasets/SA-1923-2023/AfterFeaturengineeringMainDataSA-76-23.csv'
filePathNAFZ = 'datasets/NorthAnatolian-1923-2023/AfterFeaturengineeringMainDataNAFZ-76-23.csv'

preSA = pd.read_csv(filePathSA) #Train
preNAFZ = pd.read_csv(filePathNAFZ) #Test

preSA = preSA.drop(["time"], axis=1)
preNAFZWithTime = preNAFZ
preNAFZ = preNAFZ.drop(["time"], axis=1)

preSAY = preSA[["mag_max"]]
preNAFZY = preNAFZ[["mag_max"]]
preSAX = preSA.drop(["mag_max"], axis=1)
preNAFZX = preNAFZ.drop(["mag_max"], axis=1)

# SCALING
#sc = RobustScaler() # aykırı değerlere karşı dayanıklı
sc = MinMaxScaler()

x_train = preSAX
x_test = preNAFZX
y_train = preSAY
y_test = preNAFZY

print("x_train", x_train)
print("x_test", x_test)
print("y_train", y_train)
print("y_test", y_test)

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)

y_train = np.ravel(y_train)
train_y = np.array(y_train).astype(int)

# GRIDSEARCHCV İLE EN İYİ PARAMETRE DEĞERLERİNİ BULMA

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': [0.1, 1, 10]
}

grid_search = GridSearchCV(estimator=svm.SVR(), param_grid=param_grid)
print("dsfghjkljhgf")
grid_search.fit(x_train, y_train)

print("En iyi parametreler:", grid_search.best_params_) #En iyi parametreler: {'C': 10, 'gamma': 1, 'kernel': 'rbf'}

svr = svm.SVR(kernel='rbf', C= 0.1, gamma= 1) # GridSearchCV ile optimum parametreler bulundu. MSE=0.003, MAE=0.04

svr.fit(x_train, y_train)
y_pred_svr = svr.predict(x_test) # y_pred

mae = mean_absolute_error(y_test, y_pred_svr)
mse = mean_squared_error(y_test, y_pred_svr)

print("mae",mae,"mse",mse)

weight = x_test[: , 13] #Weight(Mw) kolonunu al
print(len(weight))
print(weight)
errors  = y_test - y_pred_svr

weightedAverage = np.sum(np.abs(errors) * weight) / np.sum(weight) * 100
print("%thresholdagirlik" , weightedAverage)
print("weight" , len(weight))
weightedAverage = weightedAverage / len(weight) # len(errors)'da kullanılabilir zaten iki dizinin boyutu aynı (741)
print("thresholdagirlik" , weightedAverage) #threshold değeri

errors = y_test - y_pred_svr
standart_sapma_hatalar = np.std(errors)

# Gerçek değerlerin standart sapmasını hesaplayın
standart_sapma_gercek = np.std(y_test)

# Hata yüzdesini hesaplayın
hata_yuzdesi = (standart_sapma_hatalar / standart_sapma_gercek) * 100
print("thresholdstsapma", hata_yuzdesi)

percentage_residuals = []
residuals = y_test - y_pred_svr
absolute_residuals = np.abs(residuals) #mutlak değer ile farkını al

for index in range(len(y_test)):
    if y_test[index] != 0:
      percentage_residuals = (absolute_residuals[index] / y_test[index]) * 100
threshold = np.sum(percentage_residuals) / len(y_test)
print("thresholdbasitort", threshold )

weight = x_test[: , 13] #Weight(Mw) kolonunu al
weightedAverage=[]
errors= []
for i in range(len(y_test)):
    errors.append(y_test[i] - y_pred_svr[i])
    weightedAverage.append(np.sum(np.abs(errors[i]) * weight[i]) / weight[i] * 100)
    weightedAverage[i] = weightedAverage[i]
    if( weightedAverage[i] > 0.11):
        mainStage1.append(preNAFZWithTime.loc[i])
    else:
        mainStage2.append(preNAFZWithTime.loc[i])
#print("weigthedaverage", len(weightedAverage))
#print("mainStage1", (mainStage1))
#print("mainStage2", len(mainStage2))

filePath1 = "datasets/NorthAnatolian-1923-2023/" + "MainStage1" + ".csv"
mainStage1 = pd.DataFrame(mainStage1)
mainStage1.to_csv(filePath1)
filePath2 = "datasets/NorthAnatolian-1923-2023/" + "MainStage2" + ".csv"
mainStage2 = pd.DataFrame(mainStage2)
mainStage2.to_csv(filePath2)
"""
# 4.2. MODELLERİ EĞİTME
# 4.2.1. VERİ SETİ 1 (TÜM VERİLER) İÇİN MODEL EĞİTME
### 4.2.1.1 SVM ###
"""
filePathNAFZ = 'datasets/NorthAnatolian-1923-2023/AfterFeaturengineeringMainDataNAFZ-76-23.csv'

preNAFZ = pd.read_csv(filePathNAFZ)
preNAFZ = preNAFZ.drop(["time"], axis=1)

magM = preNAFZ[["mag_max"]] #y

# SCALING
# sc = RobustScaler() # aykırı değerlere karşı dayanıklı
sc = MinMaxScaler()
x = preNAFZ.drop(["mag_max"], axis=1)

# Veri setini eğitim, test ve doğrulama olarak ayırın
x_train, x_temp, y_train, y_temp = train_test_split(x, magM, test_size=0.3, shuffle=False) # Eğitim verisi ve geçici veri (test + doğrulama)

# Geçici veriyi test ve doğrulama olarak ayırın
x_test, x_valid, y_test, y_valid = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=False) # Test verisi ve doğrulama verisi

print("x_train", x_train)
print("x_test", x_test)
print("y_train", y_train)
print("y_test", y_test)

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)


# GRIDSEARCHCV İLE EN İYİ PARAMETRE DEĞERLERİNİ BULMA

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': [0.1, 1, 10]
}

grid_search = GridSearchCV(estimator=svm.SVR(), param_grid=param_grid)
grid_search.fit(x_train, y_train)
print("En iyi parametreler:", grid_search.best_params_) #En iyi parametreler: {'C': 10, 'gamma': 1, 'kernel': 'rbf'}

svr = svm.SVR(kernel='rbf', C= 10, gamma= 0.1) # GridSearchCV ile optimum parametreler bulundu. MSE 0.009151399941817176, MAE 0.08548841651941114
start_time = time.time()
svr.fit(x_train, y_train)
end_time = time.time()
training_time = end_time - start_time
"""
"""
y_valid_pred = svr.predict(x_valid)
mae = mean_absolute_error(y_valid, y_valid_pred)
mse = mean_squared_error(y_valid, y_valid_pred)

print("mae-val",mae,"mse-val",mse)
"""
"""
y_pred_svr = svr.predict(x_test) #y_pred

mae = mean_absolute_error(y_test, y_pred_svr)
mse = mean_squared_error(y_test, y_pred_svr)

print("SVR-MAE",mae)
print("SVR-MSE", mse)
print("SVM- Training-Time", training_time)
"""
#filePathNAFZ = 'datasets/NorthAnatolian-1923-2023/MainStage2.csv' #eşik değerinden küçükler
#filePathNAFZ = 'datasets/NorthAnatolian-1923-2023/MainStage1.csv' #eşik değerinden büyükler
#filePathNAFZ = 'datasets/NorthAnatolian-1923-2023/AfterFeaturengineeringMainDataNAFZ-76-23.csv' #tüm değerler

KNNA = []
DTRA = []
XGBA = []
SVMA=[]
RFRA=[]
filePathNAFZ = ["datasets/NorthAnatolian-1923-2023/AfterFeaturengineeringMainDataNAFZ-76-23.csv","datasets/NorthAnatolian-1923-2023/MainStage1.csv", "datasets/NorthAnatolian-1923-2023/MainStage2.csv"]
for i in filePathNAFZ:
    preNAFZ = pd.read_csv(i)
    preNAFZ["time"] = pd.to_datetime(preNAFZ["time"])
    #preNAFZ["timestamps"] = preNAFZ["time"]
    preNAFZ.set_index('time', inplace = True)
    #print("prenafz", preNAFZ )
    n = len(preNAFZ)
    train = preNAFZ[0:int(n*0.67)]
    test = preNAFZ[int(n*0.67):]

    print("train" , train)
    print("test" , test)

    f_columns = ['event_count', 'mag_mean', 'mag_sum',
           'longitude_mean', 'latitude_mean',  'depth_mean', 'time_diff_float_mean',
           'time_diff_float_std', 'depth_std', 'latitude_std', 'longitude_std',
            'mag_scatter', 'weight_richter', 'weight_Mw']

    f_transformer = MinMaxScaler()
    mag_transformer = MinMaxScaler()

    f_transformer = f_transformer.fit(train[f_columns].to_numpy())
    mag_transformer = mag_transformer.fit(train[["mag_max"]])

    f_transformer.get_params()

    train.loc[:,f_columns] = f_transformer.transform(train[f_columns].to_numpy())
    train["mag_max"] = mag_transformer.transform(train[["mag_max"]])

    test.loc[:,f_columns] = f_transformer.transform(test[f_columns].to_numpy())
    test["mag_max"] = mag_transformer.transform(test[["mag_max"]])
    def create_dataset(X, y, time_steps):
        Xs, ys = [], []
        for i in range(len(X)- time_steps - 1):
            v = X.iloc[i: (i + time_steps)].to_numpy()
            Xs.append(v)
            ys.append(y.iloc[i+time_steps])
        return np.array(Xs), np.array(ys)

    TIME_STEPS = 1 #window_size

    X_train, y_train = create_dataset(train, train["mag_max"], time_steps = TIME_STEPS)
    X_test, y_test = create_dataset(test, test["mag_max"], time_steps= TIME_STEPS)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

    num_samples, num_time_steps, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(num_samples, -1)
    num_samples, num_time_steps, num_features = X_test.shape
    X_test_reshaped = X_test.reshape(num_samples, -1)

    def XGBoostingModel():

        model = xg.XGBRegressor(
        )
        start_time = time.time()
        history=model.fit(X_train_reshaped, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        y_pred_svr = model.predict(X_test_reshaped)  # y_pred

        mae = mean_absolute_error(y_test, y_pred_svr)
        mse = mean_squared_error(y_test, y_pred_svr)
        rmse =  mean_squared_error(y_test, y_pred_svr,squared = False)
        mape = mean_absolute_percentage_error(y_test, y_pred_svr)

        rsquare = r2_score(y_test, y_pred_svr)

        print("MAE - XGBOOST", mae)
        print("MSE - XGBOOST", mse)
        print("MAPE - XGBOOST", mape)
        print("RMSE - XGBOOST", rmse)

        print("rsquare - XGBOOST", rsquare)
        print("Training_time", training_time)
        return history
    def SVMModel():

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': [0.1, 1, 10]
        }

        grid_search = GridSearchCV(estimator=svm.SVR(), param_grid=param_grid)
        num_samples, num_time_steps, num_features = X_train.shape
        X_train_reshaped = X_train.reshape(num_samples, -1)
        num_samples, num_time_steps, num_features = X_test.shape
        X_test_reshaped = X_test.reshape(num_samples, -1)

        grid_search.fit(X_train_reshaped, y_train)
        print("En iyi parametreler:",
              grid_search.best_params_)  # En iyi parametreler: {'C': 10, 'gamma': 1, 'kernel': 'rbf'}

        svr = svm.SVR(kernel='linear', C=10,
                      gamma=0.1)  # GridSearchCV ile optimum parametreler bulundu. MSE 0.009151399941817176, MAE 0.08548841651941114
        start_time = time.time()
        history=svr.fit(X_train_reshaped, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        y_pred_svr = svr.predict(X_test_reshaped)  # y_pred

        mae = mean_absolute_error(y_test, y_pred_svr)
        mse = mean_squared_error(y_test, y_pred_svr)
        rmse = mean_squared_error(y_test, y_pred_svr, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_pred_svr)

        rsquare = r2_score(y_test, y_pred_svr)

        print("MAE - SVR", mae)
        print("MSE - SVR", mse)
        print("RMSE - SVR", rmse)
        print("MAPE - SVR", mape)

        print("rsquare - SVR", rsquare)
        print("Training_time", training_time)
        return history
    def KNNRegModel():
        knn_reg = KNeighborsRegressor(n_neighbors=5)  # K sayısını isteğinize göre ayarlayabilirsiniz
        start_time = time.time()
        KNNRegPreds=knn_reg.fit(X_train_reshaped, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        # Test verileri üzerinde tahminler yapın
        y_test_pred = knn_reg.predict(X_test_reshaped)

        # Eğitim ve test hatalarını hesaplayın

        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_test_pred)
        rsquare = r2_score(y_test, y_test_pred)

        print("Test MSE - KNNREG:", test_mse)
        print("Test MAE - KNNREG:", test_mae)
        print("Test RMSE - KNNREG:", rmse)
        print("MAPE - KNNREG", mape)
        print("rsquare - KNNREG", rsquare)
        print("Training_time", training_time)
        return KNNRegPreds
    def DTRegModel():
        regressor = DecisionTreeRegressor(max_depth=5)  # Maksimum ağaç derinliğini isteğinize göre ayarlayabilirsiniz
        start_time = time.time()
        DTRegModel= regressor.fit(X_train_reshaped, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        # Tahminler için yeni X değerleri oluşturun
        X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

        # Karar ağacı modeli ile tahminler yapın
        y_test_pred = regressor.predict(X_test_reshaped)
        test_mse = mean_squared_error(y_test, y_test_pred)
        rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        mape = mean_absolute_percentage_error(y_test, y_test_pred)

        rsquare = r2_score(y_test, y_test_pred)

        print("Test MSE - DTREG:", test_mse)
        print("Test MAE - DTREG:", test_mae)
        print("Test RMSE - DTREG:", rmse)
        print("MAPE - DTREG", mape)

        print("rsquare - DTREG", rsquare)
        print("Training_time", training_time)
        return DTRegModel

    def RFRegModel():
        rf_regressor = RandomForestRegressor(n_estimators=100)
        start_time = time.time()
        RFRegModel=rf_regressor.fit(X_train_reshaped, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        # Tahminler yapma
        y_test_pred = rf_regressor.predict(X_test_reshaped)

        # Test verileri üzerinde performansı değerlendi
        # mse = mean_squared_error(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_test_pred)

        rsquare = r2_score(y_test, y_test_pred)

        print("Test MSE - RFREG:", test_mse)
        print("Test MAE - RFREG:", test_mae)
        print("Test RMSE - RFREG:", rmse)

        print("MAPE - RFREG", mape)

        print("rsquare - RFREG", rsquare)
        print("Training_time", training_time)
        return RFRegModel

    def LSTMModel(units1,dropout_rate):
        model = Sequential()
        # Adding bi-directional layer
        model.add(Bidirectional( LSTM(units1,input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=False)))
        # Adding dropout layer to regularize complexities
        model.add(Dropout(dropout_rate))
        #model.add(Bidirectional (LSTM(units2,return_sequences=False)))
        #model.add(Dropout(dropout_rate))
        # Add output layer
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam()
        # Compiling the model
        model.compile(loss = "mean_squared_error", optimizer = optimizer,metrics=['mse','mae','acc'])
        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_split=0.33, shuffle=False,
                          callbacks=[early_stopping])
        lstm_preds = model.predict(X_test)
        end_time = time.time()
        training_time = end_time - start_time

        test_mse = mean_squared_error(y_test, lstm_preds)
        test_mae = mean_absolute_error(y_test, lstm_preds)
        rmse = mean_squared_error(y_test, lstm_preds, squared=False)
        mape = mean_absolute_percentage_error(y_test, lstm_preds)

        rsquare = r2_score(y_test, lstm_preds)

        print("Test MSE - LSTM:", test_mse)
        print("Test MAE - LSTM:", test_mae)
        print("Test RMSE - LSTM:", rmse)
        print("MAPE - LSTM", mape)

        print("rsquare - LSTM", rsquare)

        """
        print("Accuracy:", np.mean(history.history['acc']))
        print("MSE:", np.mean(history.history['mse']))
        print("MAE:", np.mean(history.history['mae']))
        """
        print("Training_Time:", training_time)

        model.summary()

        # graphics
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="test")
        plt.legend()
        plt.show()
        return history

    def CNNModel():
        # Model oluşturma
        model = Sequential()

        model.add(Conv1D(filters=32, kernel_size=3, activation='sigmoid', input_shape=(X_train.shape[1],X_train.shape[2]),padding = 'same' ))

        # MaxPooling2D katmanları ekleme
        model.add(MaxPooling1D(2, padding='same'))
        # Veriyi düzleştirme
        model.add(Flatten())

        # Tam bağlantılı katmanlar ekleme (regresyon için)
        model.add(Dense(1, activation='sigmoid'))


        optimizer = Adam()
        model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse', 'mae', 'acc'])
        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_split=0.33, shuffle=False,
                            callbacks=[early_stopping])
        cnn_preds = model.predict(X_test)
        end_time = time.time()
        training_time = end_time - start_time
        # metrics

        test_mse = mean_squared_error(y_test, cnn_preds)
        test_mae = mean_absolute_error(y_test, cnn_preds)
        rmse = mean_squared_error(y_test, cnn_preds, squared=False)
        mape = mean_absolute_percentage_error(y_test, cnn_preds)

        rsquare = r2_score(y_test, cnn_preds)

        print("Test MSE - LSTM:", test_mse)
        print("Test MAE - LSTM:", test_mae)
        print("Test rmse - LSTM:", rmse)
        print("MAPE - LSTM", mape)

        print("rsquare - LSTM", rsquare)
        """
        print("Accuracy:", np.mean(history.history['acc']))
        print("MSE:", np.mean(history.history['mse']))
        print("MAE:", np.mean(history.history['mae']))
        """
        # training time
        print("Training_Time:", training_time)

        model.summary()

        # graphics
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="test")

        plt.legend()
        plt.show()
        return history

    def ARIMAModel():
        def ad_test(dataset):
            dftest = adfuller(dataset, autolag='AIC')

            print("1. ADF : ", dftest[0])
            print("2. P-Value : ", dftest[1])
            print("3. Num Of Lags : ", dftest[2])
            print("4. Num Of Observations Used For ADF Regression:", dftest[3])
            print("5. Critical Values :")
            for key, val in dftest[4].items():
                print("\t", key, ": ", val)

        ad_test(preNAFZ['mag_max'])
        stepwise_fit = auto_arima(preNAFZ['mag_max'], trace=True,
                                  suppress_warnings=True)
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(1, 1, 2))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))
            return model_fit
    ###############################################################################
    KNNRegModel = KNNRegModel()
    KNNA.append(KNNRegModel)
    DTRegModel = DTRegModel()
    DTRA.append(DTRegModel)
    xgboost_model = XGBoostingModel()
    XGBA.append(xgboost_model)
    svm_model=SVMModel()
    SVMA.append(svm_model)
    RFRegModel = RFRegModel()
    RFRA.append(RFRegModel)
    lstm_model=LSTMModel(64,0.2)
    cnn_model=CNNModel()
    #ARIMAModel()

#ENSEMBLE LEARNING

ensemble_model = VotingRegressor(estimators=[
    ('KNN1', KNNA[0]),
    ('KNN2', KNNA[1]),
    ('KNN3', KNNA[2]),
    ('DTRA1', DTRA[0]),
    ('DTRA2', DTRA[1]),
    ('DTRA3', DTRA[2]),
    ('XGBA1', XGBA[0]),
    ('XGBA2', XGBA[1]),
    ('XGBA3', XGBA[2]),
    ('SVMA1', SVMA[0]),
    ('SVMA2', SVMA[1]),
    ('SVMA3', SVMA[2]),
    ('RFRA1', RFRA[0]),
    ('RFRA2', RFRA[1]),
    ('RFRA3', RFRA[2])


] )  # Use 'soft' voting to get probabilities for the ensemble

# Fit the ensemble model
start_time = time.time()
ensemble_model.fit(X_train_reshaped, y_train)
end_time = time.time()
training_time = end_time - start_time
# Make predictions using the ensemble model
ensemble_preds = ensemble_model.predict(X_test_reshaped)

# Evaluate the ensemble model's mse
ensemble_mse = mean_squared_error(y_test, ensemble_preds)
ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
ensemble_rmse = mean_squared_error(y_test, ensemble_preds, squared=False)
ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_preds)

ensemble_rsquare = r2_score(y_test, ensemble_preds)

print("Ensemble Model Mean Squared Error:", ensemble_mse)
print("Ensemble Model Mean Absolute Error:", ensemble_mae)
print("Ensemble Model Root Mean Absolute Error:", ensemble_rmse)
print("Ensemble Model Mean Absolute Percentage Error:", ensemble_mape)

print("Ensemble Model R Square Score:", ensemble_rsquare)
print("Training Time", training_time)











