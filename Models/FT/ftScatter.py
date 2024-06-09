from scipy.fft import fft, fftfreq, ifftn, fftn, fft2, fftshift, rfft, rfftfreq
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import pandas as pd
import os
import cv2
from scipy import stats

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def spectra(img):
  n = img.shape[0]
  # We use rfft since we are processing real values
  a = rfft(img,img.shape[0], axis=0)

  # Sum power along the second axis
  a = a.real*a.real + a.imag*a.imag
  a = a.sum(axis=1)/a.shape[1]

  # Generate a list of frequencies
  f = rfftfreq(n)

  # Graph it
  #plt.plot(f[1:],a[1:], label = 'sum of amplitudes over y vs f_x')
  tempX = (signaltonoise(a[1:]))

  slope1, intercept1, r_value, p_value, std_err1 = stats.linregress(f[1:],a[1:])
  tempFinalX = np.log(a[-1])


  # Fourier Transform along the second axis

  # Same steps as above
  n = img.shape[1]

  a = rfft(img,img.shape[1],axis=1)

  a = a.real*a.real + a.imag*a.imag
  a = a.sum(axis=0)/a.shape[0]

  f = rfftfreq(n)

  #plt.plot(f[1:],a[1:],  label ='sum of amplitudes over x vs f_y')
  tempY = (signaltonoise(a[1:]))
  
  slope2, intercept2, r_value, p_value, std_err2 = stats.linregress(f[1:],a[1:])
  tempFinalY = np.log(a[-1])

  #return tempX, tempY, slope1, slope2, intercept1, intercept2, std_err1, std_err2
  return tempX, tempY, slope1, slope2, intercept1, intercept2, std_err1, std_err2 , tempFinalX, tempFinalY
  #return tempX, tempY, slope1-slope2, abs(slope2-slope1), intercept1, intercept2, std_err1, std_err2


# giving directory name
dirname = './train/Annular/'
# giving file extension
ext = ('.png')

count = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        count = count+1
    else:
        continue

print(count)


final1 = np.zeros((count,10))

# iterating over all files
i = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        img = cv2.imread(dirname + files, 0)
        x,y,s1,s2,i1,i2,d1,d2,a,b = spectra(img)
        final1[i] = [x,y,s1,s2,i1,i2,d1,d2,a,b]
        i = i + 1
    else:
        continue


final1 = pd.DataFrame(final1, columns=['x', 'y', 's1', 's2', 'i1', 'i2', 'd1', 'd2','a','b'])
final1['Group'] = 0
print(final1)

plt.scatter(final1['x'], final1['y'], s=5,alpha=0.5)

# giving directory name
dirname = './train/Bubbly/'

count = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        count = count+1
    else:
        continue

print(count)


final2 = np.zeros((count,10))

# iterating over all files
i = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        img = cv2.imread(dirname + files, 0)
        x,y,s1,s2,i1,i2,d1,d2,a,b = spectra(img)
        final2[i] = [x,y,s1,s2,i1,i2,d1,d2,a,b]
        i = i + 1
    else:
        continue


final2 = pd.DataFrame(final2, columns=['x', 'y', 's1', 's2', 'i1', 'i2', 'd1', 'd2','a','b'])
final2['Group'] = 1
print(final2)

plt.scatter(final2['x'], final2['y'], s=5,alpha=0.5)


# giving directory name
dirname = './train/ElongatedBubbly/'

count = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        count = count+1
    else:
        continue

print(count)


final3 = np.zeros((count,10))

# iterating over all files
i = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        img = cv2.imread(dirname + files, 0)
        x,y,s1,s2,i1,i2,d1,d2,a,b = spectra(img)
        final3[i] = [x,y,s1,s2,i1,i2,d1,d2,a,b]
        i = i + 1
    else:
        continue


final3 = pd.DataFrame(final3, columns=['x', 'y', 's1', 's2', 'i1', 'i2', 'd1', 'd2','a','b'])
final3['Group'] = 2
print(final3)

plt.scatter(final3['x'], final3['y'], s=5,alpha=0.5)


# giving directory name
dirname = './train/Slug/'

count = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        count = count+1
    else:
        continue

print(count)


final4 = np.zeros((count,10))

# iterating over all files
i = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        img = cv2.imread(dirname + files, 0)
        x,y,s1,s2,i1,i2,d1,d2,a,b = spectra(img)
        final4[i] = [x,y,s1,s2,i1,i2,d1,d2,a,b]
        i = i + 1
    else:
        continue


final4 = pd.DataFrame(final4, columns=['x', 'y', 's1', 's2', 'i1', 'i2', 'd1', 'd2','a','b'])
final4['Group'] = 3
print(final4)

plt.scatter(final4['x'], final4['y'], s=5,alpha=0.5)


# giving directory name
dirname = './train/StratifiedSmooth/'

count = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        count = count+1
    else:
        continue

print(count)


final5 = np.zeros((count,10))

# iterating over all files
i = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        img = cv2.imread(dirname + files, 0)
        x,y,s1,s2,i1,i2,d1,d2,a,b = spectra(img)
        final5[i] = [x,y,s1,s2,i1,i2,d1,d2,a,b]
        i = i + 1
    else:
        continue


final5 = pd.DataFrame(final5, columns=['x', 'y', 's1', 's2', 'i1', 'i2', 'd1', 'd2','a','b'])
final5['Group'] = 4
print(final5)

plt.scatter(final5['x'], final5['y'], s=5,alpha=0.5)


# giving directory name
dirname = './train/StratifiedWavy/'

count = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        count = count+1
    else:
        continue

print(count)


final6 = np.zeros((count,10))

# iterating over all files
i = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        img = cv2.imread(dirname + files, 0)
        x,y,s1,s2,i1,i2,d1,d2,a,b = spectra(img)
        final6[i] = [x,y,s1,s2,i1,i2,d1,d2,a,b]
        i = i + 1
    else:
        continue


final6 = pd.DataFrame(final6, columns=['x', 'y', 's1', 's2', 'i1', 'i2', 'd1', 'd2','a','b'])
final6['Group'] = 5
print(final6)

plt.scatter(final6['x'], final6['y'], s=5,alpha=0.5)


# giving directory name
dirname = './train/Unstable/'

count = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        count = count+1
    else:
        continue

print(count)


final7 = np.zeros((count,10))

# iterating over all files
i = 0
for files in os.listdir(dirname):
    if files.endswith(ext):
        img = cv2.imread(dirname + files, 0)
        x,y,s1,s2,i1,i2,d1,d2,a,b = spectra(img)
        final7[i] = [x,y,s1,s2,i1,i2,d1,d2,a,b]
        i = i + 1
    else:
        continue


final7 = pd.DataFrame(final7, columns=['x', 'y', 's1', 's2', 'i1', 'i2', 'd1', 'd2','a','b'])
final7['Group'] = 6
print(final7)

plt.scatter(final7['x'], final7['y'], s=5,alpha=0.5)


plt.show(block=True)

final = pd.concat([final1, final2, final3, final4, final5, final6, final7], ignore_index=True)

final.to_csv('out.csv', index=False)
