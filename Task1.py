from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def task1():
    img = ndimage.imread("tek.png", flatten=True)
    pat = ndimage.imread("fonts/arial/31.png", flatten=True)

    # img = 255 - img
    # pat = 255 - pat
    print(pat)
    plt.imshow(img)
    plt.show()

    plt.imshow(pat)
    plt.show()

    fpat = np.fft.fft2(np.rot90(pat, 2), img.shape)
    # fpat = np.fft.fft2(pat, img.shape)
    fimg = np.fft.fft2(img)
    m = np.multiply(fimg, fpat)
    corr = np.fft.ifft2(m)
    # corr = np.abs(corr)

    corr = corr.astype(float)
    corr[corr < (0.8 * np.amax(corr))] = 0
    # img[corr >= 0.6*np.amax(corr)] = 512
    plt.imshow(img)
    plt.show()

    plt.matshow(corr)
    plt.jet()
    plt.show()
    # print(corr)
    #plt.jet()

task1()

    # plot z colormap jet