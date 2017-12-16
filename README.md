
# Wind Power Generation Prediction By Weather Prediction Using RNN

### Import Libraries


```python
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as pltdt
%matplotlib inline
import seaborn as sns
sns.set_style("darkgrid")
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score
from keras.layers import Dropout
```

    Using TensorFlow backend.
    

## Data Reading, Cleaning, Mapping

### Read All Power Dataset and Weather Dataset for France in 10 Years


```python
power_countries_1986_2015 = pd.read_csv("EMHIRESPV_TSh_CF_Country_19862015.csv")
meteorological_france_2006_2015 = pd.read_csv("france_2006_2015_ver2_0_9258_487749847.csv", sep=";")
```


```python
power_countries_1986_2015.shape
```




    (262968, 29)




```python
meteorological_france_2006_2015.shape
```




    (3662956, 16)



## Power from Hours to Days, Choose 10 Yesrs of France


```python
t_h = pd.date_range('1/1/1986', periods=262968, freq='H')
power_countries_1986_2015["HOUR"] = t_h
power_countries_1986_2015.set_index("HOUR", inplace = True)
power_countries_1986_2015['DAY']=power_countries_1986_2015.index.map(lambda x: x.strftime('%Y-%m-%d'))
power_countries_1986_2015_day = power_countries_1986_2015.groupby('DAY').mean()
power_countries_2006_2015_day = power_countries_1986_2015_day.loc["2006-01-01":"2015-12-31"]
t_d = pd.date_range('1/1/2006', periods=3652, freq='D')
power_france_2006_2015_day = power_countries_2006_2015_day["FR"]
power_france_2006_2015_day.head()
```




    DAY
    2006-01-01    0.040878
    2006-01-02    0.050066
    2006-01-03    0.046397
    2006-01-04    0.061315
    2006-01-05    0.042258
    Name: FR, dtype: float64



## Aggregate Weather Dataset for Whole France


```python
meteorological_france_2006_2015_day = meteorological_france_2006_2015.groupby('DAY').mean()
meteorological_france_2006_2015_day["DAY"] = t_d
meteorological_france_2006_2015_day.set_index("DAY", inplace=True)
meteorological_france_2006_2015_day = \
    meteorological_france_2006_2015_day.\
        drop(meteorological_france_2006_2015_day.columns[[0, 1, 2, 3, -1]], axis=1) 
print(meteorological_france_2006_2015_day.head())
```

                TEMPERATURE_MAX  TEMPERATURE_MIN  TEMPERATURE_AVG  WINDSPEED  \
    DAY                                                                        
    2006-01-01         7.536590         2.931306         5.234895   5.001097   
    2006-01-02         6.788734         3.453141         5.121934   3.652941   
    2006-01-03         6.067498         1.962014         4.015254   2.862612   
    2006-01-04         4.679661         0.788136         2.735593   3.160020   
    2006-01-05         2.080359        -1.576869         0.251147   2.734397   
    
                VAPOURPRESSURE  PRECIPITATION        E0       ES0       ET0  \
    DAY                                                                       
    2006-01-01        7.792114       5.722333  0.436790  0.381186  0.614905   
    2006-01-02        7.303749       0.465204  0.439681  0.380588  0.664437   
    2006-01-03        6.738744       0.682253  0.447258  0.389910  0.617697   
    2006-01-04        6.270528       0.615753  0.361645  0.308963  0.562512   
    2006-01-05        5.392353       0.974177  0.276371  0.237338  0.437378   
    
                  RADIATION  
    DAY                      
    2006-01-01  3049.951147  
    2006-01-02  3304.593220  
    2006-01-03  3235.214357  
    2006-01-04  3584.273180  
    2006-01-05  3083.100698  
    

## Data Shifting


```python
france_2006_2015_day_up1 = meteorological_france_2006_2015_day.shift(-1)
france_2006_2015_day_up2 = meteorological_france_2006_2015_day.shift(-2)
france_2006_2015_day_down1 = meteorological_france_2006_2015_day.shift(1)
france_2006_2015_day_down2 = meteorological_france_2006_2015_day.shift(2)
france_2006_2015_day_5days = pd.concat([meteorological_france_2006_2015_day,\
                                        france_2006_2015_day_up1, france_2006_2015_day_up2, \
                                        france_2006_2015_day_down1, \
                                       france_2006_2015_day_down2], axis=1)

fr_p = power_france_2006_2015_day.rename('fr')
meteorological_france_2006_2015_day_shif = france_2006_2015_day_5days

for i in range(1, 14):
    fr_p_shift = fr_p.shift(-1)
    meteorological_france_2006_2015_day_shif = \
        pd.concat([fr_p_shift, meteorological_france_2006_2015_day_shif], axis=1)
    fr_p = fr_p_shift
    
print(meteorological_france_2006_2015_day_shif.head())
```

                      fr        fr        fr        fr        fr        fr  \
    DAY                                                                      
    2006-01-01  0.116344  0.089186  0.054274  0.068600  0.076414  0.064127   
    2006-01-02  0.105823  0.116344  0.089186  0.054274  0.068600  0.076414   
    2006-01-03  0.051696  0.105823  0.116344  0.089186  0.054274  0.068600   
    2006-01-04  0.024062  0.051696  0.105823  0.116344  0.089186  0.054274   
    2006-01-05  0.024392  0.024062  0.051696  0.105823  0.116344  0.089186   
    
                      fr        fr        fr        fr     ...       \
    DAY                                                    ...        
    2006-01-01  0.047271  0.039511  0.062992  0.042258     ...        
    2006-01-02  0.064127  0.047271  0.039511  0.062992     ...        
    2006-01-03  0.076414  0.064127  0.047271  0.039511     ...        
    2006-01-04  0.068600  0.076414  0.064127  0.047271     ...        
    2006-01-05  0.054274  0.068600  0.076414  0.064127     ...        
    
                TEMPERATURE_MAX  TEMPERATURE_MIN  TEMPERATURE_AVG  WINDSPEED  \
    DAY                                                                        
    2006-01-01              NaN              NaN              NaN        NaN   
    2006-01-02              NaN              NaN              NaN        NaN   
    2006-01-03         7.536590         2.931306         5.234895   5.001097   
    2006-01-04         6.788734         3.453141         5.121934   3.652941   
    2006-01-05         6.067498         1.962014         4.015254   2.862612   
    
                VAPOURPRESSURE  PRECIPITATION        E0       ES0       ET0  \
    DAY                                                                       
    2006-01-01             NaN            NaN       NaN       NaN       NaN   
    2006-01-02             NaN            NaN       NaN       NaN       NaN   
    2006-01-03        7.792114       5.722333  0.436790  0.381186  0.614905   
    2006-01-04        7.303749       0.465204  0.439681  0.380588  0.664437   
    2006-01-05        6.738744       0.682253  0.447258  0.389910  0.617697   
    
                  RADIATION  
    DAY                      
    2006-01-01          NaN  
    2006-01-02          NaN  
    2006-01-03  3049.951147  
    2006-01-04  3304.593220  
    2006-01-05  3235.214357  
    
    [5 rows x 63 columns]
    

## The Final Dataset, Save Object for RNN


```python
meteorological_france_2006_2015_day_shif\
    = pd.concat([power_france_2006_2015_day, meteorological_france_2006_2015_day_shif], axis=1)

meteorological_france_2006_2015_day_shif.to_pickle("wind_dataset_fr_shift1218_5days")
```

# RNN Model Building and Test

## Read Saved DataFrame


```python
df = pd.read_pickle('wind_dataset_fr_shift1218_5days')
print(df.head())
```

                      FR        fr        fr        fr        fr        fr  \
    DAY                                                                      
    2006-01-01  0.040878  0.116344  0.089186  0.054274  0.068600  0.076414   
    2006-01-02  0.050066  0.105823  0.116344  0.089186  0.054274  0.068600   
    2006-01-03  0.046397  0.051696  0.105823  0.116344  0.089186  0.054274   
    2006-01-04  0.061315  0.024062  0.051696  0.105823  0.116344  0.089186   
    2006-01-05  0.042258  0.024392  0.024062  0.051696  0.105823  0.116344   
    
                      fr        fr        fr        fr     ...       \
    DAY                                                    ...        
    2006-01-01  0.064127  0.047271  0.039511  0.062992     ...        
    2006-01-02  0.076414  0.064127  0.047271  0.039511     ...        
    2006-01-03  0.068600  0.076414  0.064127  0.047271     ...        
    2006-01-04  0.054274  0.068600  0.076414  0.064127     ...        
    2006-01-05  0.089186  0.054274  0.068600  0.076414     ...        
    
                TEMPERATURE_MAX  TEMPERATURE_MIN  TEMPERATURE_AVG  WINDSPEED  \
    DAY                                                                        
    2006-01-01              NaN              NaN              NaN        NaN   
    2006-01-02              NaN              NaN              NaN        NaN   
    2006-01-03         7.536590         2.931306         5.234895   5.001097   
    2006-01-04         6.788734         3.453141         5.121934   3.652941   
    2006-01-05         6.067498         1.962014         4.015254   2.862612   
    
                VAPOURPRESSURE  PRECIPITATION        E0       ES0       ET0  \
    DAY                                                                       
    2006-01-01             NaN            NaN       NaN       NaN       NaN   
    2006-01-02             NaN            NaN       NaN       NaN       NaN   
    2006-01-03        7.792114       5.722333  0.436790  0.381186  0.614905   
    2006-01-04        7.303749       0.465204  0.439681  0.380588  0.664437   
    2006-01-05        6.738744       0.682253  0.447258  0.389910  0.617697   
    
                  RADIATION  
    DAY                      
    2006-01-01          NaN  
    2006-01-02          NaN  
    2006-01-03  3049.951147  
    2006-01-04  3304.593220  
    2006-01-05  3235.214357  
    
    [5 rows x 64 columns]
    

## Training(2006-2014) and Test(2015)


```python
df.index = pd.to_datetime(df.index)
df.index
df = df.loc["2006-01-03":"2015-12-18"]
all_y = df['FR'].values
all_X = df.drop('FR', axis=1).values
train_df = df.loc["2006-01-03":"2014-12-31"]
test_df = df.loc["2015-01-01":"2015-12-18"]
train_y = train_df['FR'].values
train_X = train_df.drop('FR', axis=1).values
test_y = test_df['FR'].values
text_X = test_df.drop('FR', axis=1).values
X_all = preprocessing.scale(all_X)
X_train = preprocessing.scale(train_X)
X_test = preprocessing.scale(text_X)
```

## Save Feathers Plot for Report as X.png


```python
ax = pd.DataFrame(X_all).plot\
    (legend='reverse', alpha=1, title="Weather Feathers", \
     kind="line", colormap="prism", figsize=(25, 10), linewidth=1, linestyle=':')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 7})
fig = ax.get_figure()
fig.savefig('X.png') 
```

## Save Power Plot for Report as Y.png


```python
df["FR"].plot(title="Power Generation", kind="line", colormap="summer", figsize=(25, 10))
ax = test_df["FR"].plot(title="Power Generation", kind="line", colormap="flag", figsize=(25, 10))
fig = ax.get_figure()
fig.savefig('Y.png') 
```

## Re-dimensioned for LSTM layer


```python
X_train_w = X_train.reshape(train_X.shape[0], 1, 63)
X_test_w = X_test.reshape(text_X.shape[0], 1, 63)
X_train_w.shape
```




    (3285, 1, 63)



## Build the Model


```python
K.clear_session()

in_sh = (1, 63) 
hidden_1= 2000
hidden_2= 2000
hidden_3= 2000
hidden_4= 2000
hidden_5= 2000
hidden_6= 2000
hidden_7= 2000
hidden_8= 2000
hidden_9= 2000
hidden_10= 2000
hidden_11= 2000
hidden_12= 2000
hidden_13= 2000
hidden_14= 2000
hidden_15= 2000
hidden_16= 2000
hidden_17= 2000
hidden_18= 2000
hidden_19= 2000
outputs = 1

model = Sequential()
model.add(LSTM(hidden_1, input_shape = in_sh,))
model.add(Dense(hidden_2, activation = 'relu'))
model.add(Dense(hidden_3, activation = 'relu'))
model.add(Dense(hidden_4, activation = 'relu'))
model.add(Dense(hidden_5, activation = 'relu'))
model.add(Dense(hidden_6, activation = 'relu'))
# model.add(Dropout(0.23))
model.add(Dense(hidden_7, activation = 'relu'))
model.add(Dense(hidden_8, activation = 'relu'))
model.add(Dense(hidden_9, activation = 'relu'))
model.add(Dense(hidden_10, activation = 'relu'))
# model.add(Dense(hidden_11, activation = 'relu'))
# model.add(Dense(hidden_12, activation = 'relu'))
# model.add(Dense(hidden_13, activation = 'relu'))
# model.add(Dense(hidden_14, activation = 'relu'))
# model.add(Dense(hidden_15, activation = 'relu'))
# model.add(Dense(hidden_16, activation = 'relu'))
# model.add(Dense(hidden_17, activation = 'relu'))
# model.add(Dense(hidden_18, activation = 'relu'))
# model.add(Dense(hidden_19, activation = 'relu'))
model.add(Dense(outputs))
model.compile(optimizer='adam', loss='mean_squared_error',)
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 2000)              16512000  
    _________________________________________________________________
    dense_1 (Dense)              (None, 2000)              4002000   
    _________________________________________________________________
    dense_2 (Dense)              (None, 2000)              4002000   
    _________________________________________________________________
    dense_3 (Dense)              (None, 2000)              4002000   
    _________________________________________________________________
    dense_4 (Dense)              (None, 2000)              4002000   
    _________________________________________________________________
    dense_5 (Dense)              (None, 2000)              4002000   
    _________________________________________________________________
    dense_6 (Dense)              (None, 2000)              4002000   
    _________________________________________________________________
    dense_7 (Dense)              (None, 2000)              4002000   
    _________________________________________________________________
    dense_8 (Dense)              (None, 2000)              4002000   
    _________________________________________________________________
    dense_9 (Dense)              (None, 2000)              4002000   
    _________________________________________________________________
    dense_10 (Dense)             (None, 1)                 2001      
    =================================================================
    Total params: 52,532,001
    Trainable params: 52,532,001
    Non-trainable params: 0
    _________________________________________________________________
    

## Set Stop Params


```python
early_stop = EarlyStopping(monitor = 'loss', patience = 10, verbose = 1)
```

## Run Model and Settings


```python
eps = 500
bs = 300

history \
    = model.fit(X_train_w, train_y, epochs = eps,\
                batch_size = bs, verbose = 1 , callbacks = [early_stop])
```

    Epoch 1/500
    3285/3285 [==============================] - 7s 2ms/step - loss: 2.1329
    Epoch 2/500
    3285/3285 [==============================] - 1s 337us/step - loss: 0.0037
    Epoch 3/500
    3285/3285 [==============================] - 1s 338us/step - loss: 0.0013
    Epoch 4/500
    3285/3285 [==============================] - 1s 338us/step - loss: 0.0011
    Epoch 5/500
    3285/3285 [==============================] - 1s 338us/step - loss: 0.0010
    Epoch 6/500
    3285/3285 [==============================] - 1s 337us/step - loss: 9.9924e-04
    Epoch 7/500
    3285/3285 [==============================] - 1s 338us/step - loss: 9.7966e-04
    Epoch 8/500
    3285/3285 [==============================] - 1s 336us/step - loss: 9.4509e-04
    Epoch 9/500
    3285/3285 [==============================] - 1s 338us/step - loss: 8.8813e-04
    Epoch 10/500
    3285/3285 [==============================] - 1s 338us/step - loss: 7.6763e-04
    Epoch 11/500
    3285/3285 [==============================] - 1s 338us/step - loss: 6.6596e-04
    Epoch 12/500
    3285/3285 [==============================] - 1s 337us/step - loss: 6.7669e-04
    Epoch 13/500
    3285/3285 [==============================] - 1s 339us/step - loss: 5.8055e-04
    Epoch 14/500
    3285/3285 [==============================] - 1s 337us/step - loss: 5.4667e-04
    Epoch 15/500
    3285/3285 [==============================] - 1s 337us/step - loss: 4.6217e-04
    Epoch 16/500
    3285/3285 [==============================] - 1s 338us/step - loss: 4.3430e-04
    Epoch 17/500
    3285/3285 [==============================] - 1s 338us/step - loss: 4.5242e-04
    Epoch 18/500
    3285/3285 [==============================] - 1s 336us/step - loss: 4.0871e-04
    Epoch 19/500
    3285/3285 [==============================] - 1s 337us/step - loss: 3.8575e-04
    Epoch 20/500
    3285/3285 [==============================] - 1s 336us/step - loss: 3.5401e-04
    Epoch 21/500
    3285/3285 [==============================] - 1s 336us/step - loss: 3.1266e-04
    Epoch 22/500
    3285/3285 [==============================] - 1s 338us/step - loss: 2.9282e-04
    Epoch 23/500
    3285/3285 [==============================] - 1s 338us/step - loss: 4.8642e-04
    Epoch 24/500
    3285/3285 [==============================] - 1s 337us/step - loss: 4.8540e-04
    Epoch 25/500
    3285/3285 [==============================] - 1s 338us/step - loss: 3.1773e-04
    Epoch 26/500
    3285/3285 [==============================] - 1s 338us/step - loss: 2.8247e-04
    Epoch 27/500
    3285/3285 [==============================] - 1s 339us/step - loss: 3.2671e-04
    Epoch 28/500
    3285/3285 [==============================] - 1s 346us/step - loss: 2.9463e-04
    Epoch 29/500
    3285/3285 [==============================] - 1s 343us/step - loss: 2.6084e-04
    Epoch 30/500
    3285/3285 [==============================] - 1s 338us/step - loss: 2.6250e-04
    Epoch 31/500
    3285/3285 [==============================] - 1s 338us/step - loss: 2.4056e-04
    Epoch 32/500
    3285/3285 [==============================] - 1s 337us/step - loss: 2.0603e-04
    Epoch 33/500
    3285/3285 [==============================] - 1s 340us/step - loss: 2.0531e-04
    

## Save Loss for Report (loss1.png)


```python
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss1.png')
```


![png](data_rnn_organized_files/data_rnn_organized_34_0.png)


## Accuracy for Test


```python
pred_y = model.predict(X_test_w)
explained_variance_score(test_y, pred_y)
```




    0.9587285287806383



## Re-train


```python
early_stop = EarlyStopping(monitor = 'loss', patience = 1, verbose = 1)
eps = 500
bs = 3287

history \
    = model.fit(X_train_w, train_y, epochs = eps,\
                batch_size = bs, verbose = 1 , callbacks = [early_stop])
```

    Epoch 1/500
    3285/3285 [==============================] - 0s 129us/step - loss: 1.1458e-04
    Epoch 2/500
    3285/3285 [==============================] - 0s 122us/step - loss: 8.2621e-05
    Epoch 3/500
    3285/3285 [==============================] - 0s 120us/step - loss: 9.7137e-05
    Epoch 00003: early stopping
    


```python
pred_y = model.predict(X_test_w)
explained_variance_score(test_y, pred_y)
```




    0.96115400680277352



# Accuracy Can be Improvment by Optimazing Params, but overall will >96%

## Save Prediction for Report (foo#.png)


```python
plt.figure(figsize=(25,8))
plt.plot(test_y, color='cyan', linewidth=4)
plt.plot(pred_y, color='red', linewidth=4, linestyle='dashed')

plt.savefig('foo3.png')
```
