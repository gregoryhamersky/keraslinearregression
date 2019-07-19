import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt



stimall,avgResp = strf_tensor(mfilename,cellid,fs)

data = stimall
label = avgResp

def base_model():
    model = Sequential()
    model.add(Dense(1,input_shape=(data.shape[1],),kernel_initializer='normal',activation='linear'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model




model = base_model()

model.fit(data,label,batch_size=data.shape[0],epochs=25000,verbose=1)



fitvals = model.layers[0].get_weights()[0].squeeze()
fitH = np.reshape(fitvals,(15,25),order='F')

plt.figure()
plt.imshow(fitH)
plt.title("fitH")



X = np.swapaxes(stimall,0,1)
XT = stimall
Y = np.swapaxes(np.expand_dims(avgResp,axis=1),0,1)
C = np.dot(X,XT)
Cinv = np.linalg.pinv(C)

Htemp = np.dot(np.dot(Y,XT),Cinv)
H = np.reshape(Htemp,(15,25),order='F')

plt.figure()
plt.imshow(H)
plt.title("H")


plt.figure()
plt.scatter(Hmess.ravel(), fitHmess.ravel())
plt.xlabel('linalg')
plt.ylabel('fitted values')

['AMT005c-02-1',
 'AMT005c-02-2',
 'AMT005c-04-1',
 'AMT005c-08-1',
 'AMT005c-11-1',
 'AMT005c-12-1',
 'AMT005c-13-1',
 'AMT005c-13-2',
 'AMT005c-14-1',
 'AMT005c-19-1',
 'AMT005c-20-1']