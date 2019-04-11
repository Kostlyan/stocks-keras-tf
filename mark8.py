from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import talib
from zigzag import *
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.utils import np_utils
from keras.callbacks import History
from keras.callbacks import TensorBoard
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

timestr = time.strftime("%Y%m%d-%H%M%S")
scaler = StandardScaler()

df1 = read_csv('SBER_170101_190228_10min.csv',
               parse_dates=[['<DATE>', '<TIME>']])  # заливаем csv файл с коировками, скаченными с finam

a1 = 200  # переменная, которая задает количество прогнозов, которые мыхотим получить, можно так же

prevclose = df1['<CLOSE>']  # n-1
prevclose = pd.Series(prevclose, dtype=float)
prevclose = prevclose[:-1]
prevclose = prevclose.reset_index()
prevclose = prevclose['<CLOSE>']

nextclose = df1['<CLOSE>']
nextclose = pd.Series(nextclose, dtype=float)
nextclose = nextclose[2:]
nextclose = nextclose.reset_index()
nextclose = nextclose['<CLOSE>']

df1 = df1[1:-1]
df1 = df1.reset_index()

df1['prevclose'] = pd.Series(prevclose)
df1['nextclose'] = pd.Series(nextclose)

rsi = []
rsi = talib.RSI(df1['<CLOSE>'].values, timeperiod=9)
df1['rsi'] = pd.Series(rsi)

sma = []
sma = talib.SMA(df1['<CLOSE>'].values, timeperiod=9)
df1['sma'] = pd.Series(sma)

ema = []
ema = talib.EMA(df1['<CLOSE>'].values, timeperiod=12)
df1['ema'] = pd.Series(ema)

roc = []
roc = talib.ROC(df1['<CLOSE>'].values, timeperiod=12)
df1['roc'] = pd.Series(roc)

sar = []
high999 = talib.MAX(df1['<CLOSE>'].values, timeperiod=20)
low999 = talib.MIN(df1['<CLOSE>'].values, timeperiod=20)
sar = talib.SAR(high999, low999, acceleration=0.02, maximum=0.2)
df1['sar'] = pd.Series(sar)

STDDEV = []
STDDEV = talib.STDDEV(df1['<CLOSE>'].values, timeperiod=9)
df1['STDDEV'] = pd.Series(STDDEV)

WMA = []
WMA = talib.WMA(df1['<CLOSE>'].values, timeperiod=9)
df1['WMA'] = pd.Series(WMA)

WILLR = []
WILLR = talib.WILLR(df1['<HIGH>'].values, df1['<LOW>'].values, df1['<CLOSE>'].values, timeperiod=9)
df1['WILLR'] = pd.Series(WILLR)

ATR = []
ATR = talib.ATR(df1['<HIGH>'].values, df1['<LOW>'].values, df1['<CLOSE>'].values, timeperiod=9)
df1['ATR'] = pd.Series(ATR)

prst_prevcl_cl = df1['<CLOSE>'] / df1['prevclose']  - 1
df1['prst_prevcl_cl'] = pd.Series(prst_prevcl_cl)

prst_cl_nextcl = df1['nextclose'] / df1['<CLOSE>'] - 1
df1['prst_cl_nextcl'] = pd.Series(prst_cl_nextcl)

high955 = talib.MAX(df1['<CLOSE>'].values, timeperiod=6)
low955 = talib.MIN(df1['<CLOSE>'].values, timeperiod=6)
zz = peak_valley_pivots(df1['<CLOSE>'].values, 0.005, -0.005)

df1['high955'] = pd.Series(high955)
df1['low955'] = pd.Series(low955)
#df1['zz'] = pd.Series(zz)

label = []
df1['label'] = pd.Series(zz)
print('label')
print(df1['label'])

for y in range(2,len(df1['label'])):       #up/down marks
    if df1['label'][y] == 1 and df1['label'][y-1] != 1:
        if y > 3 and y < (len(df1['label'])-1):
            df1.loc[y+1,'label'] = 1
            df1.loc[y-1,'label'] = 1
    elif df1['label'][y] == -1 and df1['label'][y-1] != (-1):
        if y > 3 and y < (len(df1['label'])-1):
            df1.loc[y + 1,'label'] = (-1)
            df1.loc[y - 1,'label'] = (-1)

df1 = df1[100:]
df1 = df1.reset_index()

labelprev = []
for row in df1['prst_prevcl_cl']:       #up/down marks
    if row > 0:
        labelprev.append('0')
    else:
        labelprev.append('1')

df1['labelprev'] = pd.Series(labelprev)

rsi_class = []
for row in df1['rsi']:              #rsi marks
    if row > 0 and row <= 10:
        rsi_class.append('0')
    elif row > 10 and row <= 20:
        rsi_class.append('1')
    elif row > 20 and row <= 30:
        rsi_class.append('2')
    elif row > 30 and row <= 40:
        rsi_class.append('3')
    elif row > 40 and row <= 50:
        rsi_class.append('4')
    elif row > 50 and row <= 60:
        rsi_class.append('5')
    elif row > 60 and row <= 70:
        rsi_class.append('6')
    elif row > 70 and row <= 80:
        rsi_class.append('7')
    elif row > 80 and row <= 90:
        rsi_class.append('8')
    elif row > 90 and row <= 100:
        rsi_class.append('9')

df1['rsi_class'] = pd.Series(rsi_class)

sma_class = []
sma_vs_close = df1['sma'] / df1['<CLOSE>'] - 1
for row in sma_vs_close:       #sma marks
    if row > 0:
        sma_class.append('0')
    else:
        sma_class.append('1')

df1['sma_class'] = pd.Series(sma_class)

ema_class = []
ema_vs_close = df1['ema'] / df1['<CLOSE>'] -1
for row in ema_vs_close:       #ema marks
    if row > 0:
        ema_class.append('0')
    else:
        ema_class.append('1')

df1['ema_class'] = pd.Series(ema_class)

roc_class = []
for row in df1['roc']:       #ema marks
    if row > 0:
        roc_class.append('0')
    else:
        roc_class.append('1')

df1['roc_class'] = pd.Series(roc_class)

sar_class = []
sar_vs_close = df1['sar'] / df1['<CLOSE>'] -1
for row in sar_vs_close:       #ema marks
    if row > 0:
        sar_class.append('0')
    else:
        sar_class.append('1')

df1['sar_class'] = pd.Series(sar_class)

WMA_class = []
WMA_vs_close = df1['WMA'] / df1['<CLOSE>'] - 1
for row in WMA_vs_close:       #sma marks
    if row > 0:
        WMA_class.append('0')
    else:
        WMA_class.append('1')

df1['WMA_class'] = pd.Series(WMA_class)

WILLR_class = []
for row in df1['WILLR']:       #ema marks
    if row > -20:
        WILLR_class.append('0')
    elif row > -50 and row <= -20:
        WILLR_class.append('1')
    elif row > -80 and row <= -50:
        WILLR_class.append('2')
    elif row <= -80:
        WILLR_class.append('3')

df1['WILLR_class'] = pd.Series(WILLR_class)


ATR_class = []
for row in df1['ATR']:              #rsi marks
    if row > 0 and row <= 0.2:
        ATR_class.append('0')
    elif row > 0.2 and row <= 0.3:
        ATR_class.append('1')
    elif row > 0.3 and row <= 0.4:
        ATR_class.append('2')
    elif row > 0.4 and row <= 0.5:
        ATR_class.append('3')
    elif row > 0.5 and row <= 0.6:
        ATR_class.append('4')
    elif row > 0.6 and row <= 0.7:
        ATR_class.append('5')
    elif row > 0.7 and row <= 0.8:
        ATR_class.append('6')
    elif row > 0.8 and row <= 0.9:
        ATR_class.append('7')
    elif row > 0.9 and row <= 1:
        ATR_class.append('8')
    elif row > 1:
        ATR_class.append('9')

df1['ATR_class'] = pd.Series(ATR_class)


STDDEV_class = []
for row in df1['STDDEV']:              #rsi marks
    if row > 0 and row <= 0.2:
        STDDEV_class.append('0')
    elif row > 0.2 and row <= 0.4:
        STDDEV_class.append('1')
    elif row > 0.4 and row <= 0.6:
        STDDEV_class.append('2')
    elif row > 0.6 and row <= 0.8:
        STDDEV_class.append('3')
    elif row > 0.8 and row <= 1:
        STDDEV_class.append('4')
    elif row > 1 and row <= 1.2:
        STDDEV_class.append('5')
    elif row > 1.2 and row <= 1.4:
        STDDEV_class.append('6')
    elif row > 1.4 and row <= 1.6:
        STDDEV_class.append('7')
    elif row > 1.6 and row <= 1.8:
        STDDEV_class.append('8')
    elif row > 1.8:
        STDDEV_class.append('9')

df1['STDDEV_class'] = pd.Series(STDDEV_class)
df1.to_csv('df1_mark6_' + str(timestr) + '.csv')

df1 = df1.drop(df1.query('label==0').sample(frac=.90).index)
df1.to_csv('df1_mark6_' + str(timestr) + '_dropped.csv')
print(df1.head())

cols = ['WILLR_class', 'WMA_class', 'roc_class', 'ema_class', 'sma_class', 'rsi_class', 'STDDEV_class', 'ATR_class']
train_df1 = df1[cols]
#scaler.fit(train_df1)
#train_df1 = scaler.transform(train_df1)
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder_Y_1 = LabelEncoder()
labelencoder_Y_1.fit(df1['label'])
yy = labelencoder_Y_1.transform(df1['label'])
yy = to_categorical(yy)
yy_integers = labelencoder_Y_1.transform(df1['label'])   #закомментить если классы не int
print(yy[:5])

X_train = train_df1[:-a1]  # обучающая выборка
X_train = np.array(X_train).reshape((len(X_train), len(cols)))
print(X_train[:5])

X_test = train_df1[-a1:-1]  # тестовая выборка
X_test = np.array(X_test).reshape((len(X_test), len(cols)))
print(X_test[:5])

y_train = yy[:-a1]  # цель обучающей выборки
y_train = y_train.reshape((len(y_train), 3))

y_test = yy[-a1:-1]  # реальные данные соответствующие тестовой выборке, цель, которую нужно в идеале
y_test = y_test.reshape((len(y_test), 3))

class_weights = compute_class_weight('balanced', np.unique(yy_integers), yy_integers)
#class_weights = {40.35, 0.001, 40.4}
print(class_weights)

np.random.seed(1957) # для воспроизводимости результатов
# сеть и ее обучение
NB_EPOCH = 100
BATCH_SIZE = 16
VERBOSE = 1
NB_CLASSES = 3 # количество результатов = числу исходов
OPTIMIZER = Adam()
N_HIDDEN = 128


model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(len(cols), )))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
optimizer=OPTIMIZER,
metrics=['categorical_accuracy'])

#tensorboard = TensorBoard(log_dir='logs/{}'.format(time.time()))

history = History()
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, callbacks=[history])

print(history.history.keys())

score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])

pred1 = model.predict_proba(X_test)
pd_predl = pd.DataFrame(pred1)
pd_ytest = pd.DataFrame(y_test)
pd_ytest.to_csv('y_test_' + str(timestr) + '.csv')
pd_predl.to_csv('predl_' + str(timestr) + '.csv')
#pred1 = np.array(pred1).reshape((len(pred1), 1))
prediction_data = pred1[-1]

print("prediction data:")
print(prediction_data)

model.save_weights('mark8weights_' + str(timestr) + '.h5')
model_json = model.to_json()
# Записываем модель в файл
json_file = open('modelmark8_' + str(timestr) + '.json', "w")
json_file.write(model_json)
json_file.close()

# история accuracy
plt.plot(history.history['categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# история loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()