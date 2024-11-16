Date            0
series          0
OPEN            0
HIGH            0
LOW             0
PREV. CLOSE     0
ltp             0
close           0
vwap            0
52W H           0
52W L           0
VOLUME          0
VALUE           0
No of trades    0
dtype: int64
Date            2345
series          2345
OPEN            2345
HIGH            2345
LOW             2345
PREV. CLOSE     2345
ltp             2345
close           2345
vwap            2345
52W H           2345
52W L           2345
VOLUME          2345
VALUE           2345
No of trades    2345
dtype: int64
15
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2345 entries, 0 to 2344
Data columns (total 14 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   Date          2345 non-null   object
 1   series        2345 non-null   object
 2   OPEN          2345 non-null   float64
 3   HIGH          2345 non-null   float64
 4   LOW           2345 non-null   float64
 5   PREV. CLOSE   2345 non-null   float64
 6   ltp           2345 non-null   float64
 7   close         2345 non-null   float64
 8   vwap          2345 non-null   float64
 9   52W H         2345 non-null   float64
 10  52W L         2345 non-null   float64
 11  VOLUME        2345 non-null   int64
 12  VALUE         2345 non-null   float64
 13  No of trades  2345 non-null   int64
dtypes: float64(10), int64(2), object(2)
memory usage: 256.6+ KB
        Date series    OPEN    HIGH     LOW  PREV. CLOSE     ltp   close    vwap  52W H   52W L   VOLUME         VALUE  No of trades  Day  Month  Year
0 2014-10-30     EQ  353.10  357.75  351.50       354.45  354.65  354.90  355.09  387.5  307.55  9903102  3.516512e+09         91836   30     10  2014
1 2014-10-31     EQ  356.85  360.00  352.25       354.90  354.70  355.20  356.37  387.5  307.55  9849506  3.510063e+09        136371   31     10  2014
2 2014-11-03     EQ  356.90  357.65  353.20       355.20  355.50  355.45  355.22  387.5  307.55  4960915  1.762229e+09         62037    3     11  2014
3 2014-11-05     EQ  355.70  359.80  355.50       355.45  359.00  359.15  358.00  387.5  307.55  5061240  1.811930e+09         92766    5     11  2014
4 2014-11-07     EQ  358.35  360.50  353.00       359.15  355.45  355.65  355.31  387.5  307.55  6223477  2.211257e+09        106071    7     11  2014
(938, 8) (235, 8) (938,) (235,)
Decision Tree MSE: 7664.258938909287, MAE: 79.37554996776277
Linear Regression MSE: 7009.872683893106, MAE: 79.25794737952346
Random Forest MSE: 7696.05647188298, MAE: 79.1957404255319
(1641, 8) (704, 8) (1641,) (704,)
              precision    recall  f1-score   support

           0       0.31      1.00      0.47       207
           1       1.00      0.05      0.10       497

    accuracy                           0.33       704
   macro avg       0.65      0.53      0.28       704
weighted avg       0.80      0.33      0.21       704

Last few rows of predictions:
            Date     LOW    OPEN      VOLUME    HIGH   close  Day  Month  Year  Prediction  Predictions_DecisionTree  Predictions_LR  Predictions_RF
2340 2024-03-21  416.20  419.00  19227982.0  422.00  421.25   21      3  2024         NaN                379.610714      375.065919        378.2140
2341 2024-03-22  421.25  421.25  38861818.0  429.65  428.60   22      3  2024         NaN                379.610714      376.912180        378.4280
2342 2024-03-26  425.55  425.55  10274145.0  429.20  427.65   26      3  2024         NaN                379.610714      373.384261        378.8830
2343 2024-03-27  426.45  427.00  12227545.0  428.90  428.00   27      3  2024         NaN                379.610714      373.750374        378.9195
2344 2024-03-28  427.00  428.00  27933211.0  433.25  428.35   28      3  2024         NaN                379.610714      376.072133        380.4370
Prediction date range:
 2019-07-31 00:00:00 to 2024-03-28 00:00:00
2024-11-12 06:45:24.606811: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
C:\Users\drumm\AppData\Roaming\Python\Python312\site-packages\keras\src\layers\rnn\rnn.py:204: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
Epoch 1/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 3s 20ms/step - loss: 0.0279
Epoch 2/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 0.0026
Epoch 3/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 0.0025
Epoch 4/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 0.0022
Epoch 5/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0020
Epoch 6/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - loss: 0.0016
Epoch 7/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - loss: 0.0021
Epoch 8/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0019
Epoch 9/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 0.0019
Epoch 10/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0013
Epoch 11/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0014
Epoch 12/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 0.0014    
Epoch 13/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 0.0016
Epoch 14/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0013    
Epoch 15/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0012
Epoch 16/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 0.0013
Epoch 17/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0011    
Epoch 18/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0012    
Epoch 19/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0012
Epoch 20/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0011
Epoch 21/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 0.0010    
Epoch 22/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0011
Epoch 23/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 0.0014    
Epoch 24/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0010
Epoch 25/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0012    
Epoch 26/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0010    
Epoch 27/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 9.0744e-04
Epoch 28/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 9.1880e-04
Epoch 29/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0011
Epoch 30/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 8.8072e-04
Epoch 31/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 8.2847e-04
Epoch 32/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.0010    
Epoch 33/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 7.5483e-04
Epoch 34/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 7.0824e-04
Epoch 35/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 8.9523e-04
Epoch 36/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 6.9515e-04
Epoch 37/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 7.3945e-04
Epoch 38/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 7.4106e-04
Epoch 39/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 7.0507e-04
Epoch 40/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 8.0933e-04
Epoch 41/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 8.4879e-04
Epoch 42/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 7.8972e-04
Epoch 43/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 6.8773e-04
Epoch 44/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 6.0244e-04
Epoch 45/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 7.3414e-04
Epoch 46/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 6.6082e-04
Epoch 47/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 5.9809e-04
Epoch 48/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - loss: 6.0521e-04
Epoch 49/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 6.3041e-04
Epoch 50/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 24ms/step - loss: 5.1418e-04
Epoch 51/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 6.0146e-04
Epoch 52/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 6.3475e-04
Epoch 53/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 6.6924e-04
Epoch 54/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 7.5712e-04
Epoch 55/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 5.2929e-04
Epoch 56/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 5.6483e-04
Epoch 57/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 6.3645e-04
Epoch 58/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 5.3274e-04
Epoch 59/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 22ms/step - loss: 5.3963e-04
Epoch 60/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 22ms/step - loss: 5.8613e-04
Epoch 61/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 5.2061e-04
Epoch 62/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.9808e-04
Epoch 63/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 5.8416e-04
Epoch 64/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 5.1878e-04
Epoch 65/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 5.0099e-04
Epoch 66/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.2053e-04
Epoch 67/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 5.0430e-04
Epoch 68/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.0169e-04
Epoch 69/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 3.9468e-04
Epoch 70/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 3.7799e-04
Epoch 71/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 5.2315e-04
Epoch 72/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 4.2962e-04
Epoch 73/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.3852e-04
Epoch 74/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.8094e-04
Epoch 75/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 4.6288e-04
Epoch 76/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 7.2412e-04
Epoch 77/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 4.8464e-04
Epoch 78/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - loss: 4.4517e-04
Epoch 79/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 3.7320e-04
Epoch 80/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 4.8908e-04
Epoch 81/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 3.9397e-04
Epoch 82/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 5.9111e-04
Epoch 83/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.5149e-04
Epoch 84/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 3.8216e-04
Epoch 85/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 22ms/step - loss: 3.9778e-04
Epoch 86/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 4.3076e-04
Epoch 87/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 22ms/step - loss: 3.9603e-04
Epoch 88/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - loss: 3.8474e-04
Epoch 89/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.2866e-04
Epoch 90/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 3.8070e-04
Epoch 91/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 4.3663e-04
Epoch 92/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.6473e-04
Epoch 93/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.2258e-04
Epoch 94/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.1255e-04
Epoch 95/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.3992e-04
Epoch 96/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 22ms/step - loss: 4.3310e-04
Epoch 97/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 3.6914e-04
Epoch 98/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - loss: 3.6849e-04
Epoch 99/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 3.6215e-04
Epoch 100/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 4.5760e-04
15/15 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
Decision Tree Regressor MSE: 1247.0034735618037 MAE: 22.312430042744058
Linear Regression MSE: 9.124355344807185e-27 MAE: 8.271525939264186e-14
Random Forest Regressor MSE: 985.2698509425577 MAE: 18.790996717724255
LSTM Model MSE: 67.55163579461913 MAE: 6.638300014637754

Best Model based on MSE: Linear Regression