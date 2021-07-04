import numpy as np
from scipy.interpolate import interp2d


def cg(x0, A, y, i_max, tau):
    # Solve a symmetric positive definite system Ax = y via conjugate gradients.
    # A is a function handler

    x = x0
    r = y - A(x0)
    d = r
    delta = np.sum(r.flatten()*r.flatten())

    for k in range(i_max):
        if delta < tau**2:
            break
        q = A(d)
        alpha = delta/np.sum(d.flatten()*q.flatten())
        x = x + alpha*d
        r = r - alpha*q
        deltaold = delta
        delta = np.sum(r.flatten()*r.flatten())
        beta = delta/deltaold
        d = r + beta*d
        # if np.mod(k,10)==0:
        #     print("cg: finished iteration k = ", k, ", norm(residual) = ", np.sqrt(delta))
    residual = delta
    x_star = x
    iter = k
    return (x_star, iter, residual)


def cconv2_by_fft2(A,B,flag_invertB=0,epsilon=0.01):
    # assumes that A (2D image) is bigger than B (2D kernel)

    m, n = A.shape[:2]
    mb, nb = B.shape[:2]

    # pad, multiply and transform back
    bigB = np.zeros((m, n))
    bigB[:mb,:nb] = B
    bigB = np.roll(bigB, (-int((mb-1)/2), -int((mb-1)/2)), axis=(0,1))  # pad PSF with zeros to whole image domain, and center it

    fft2B = np.fft.fft2(bigB)
    if flag_invertB:
        fft2B = np.conj(fft2B) / ((np.abs(fft2B)**2) + epsilon)  # Standard Tikhonov Regularization

    return np.real(np.fft.ifft2(np.fft.fft2(A) * fft2B))


def downsample(x,K):
    return x[::K,:]

def downsample2(x,K):
    temp = downsample(x,K).T
    y = downsample(temp,K).T
    return y

def upsample(x,K):
    y = np.zeros((x.shape[0]*K,x.shape[1]))
    y[::K,:] = x
    return y

def upsample2_MN(x,K,M,N):
    temp = upsample(x,K).T
    y = upsample(temp,K).T
    if y.shape[0]>M:
        y = np.delete(y, range(M, y.shape[0]), axis=0) # make sure to keep original size after down-up
    if y.shape[1]>N:
        y = np.delete(y, range(N, y.shape[1]), axis=1)  # make sure to keep original size after down-up
    return y


def cubic(x):
    # See Keys, "Cubic Convolution Interpolation for Digital Image
    # Processing, " IEEE Transactions on Acoustics, Speech, and Signal Processing, Vol.ASSP - 29, No. 6, December 1981, p.1155.
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    f = (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2))
    return f

def prepare_cubic_filter(scale):
    # uses the kernel part of matlab's imresize function (before subsampling)
    # note: scale<1 for downsampling

    kernel_width = 4
    kernel_width = kernel_width / scale

    u = 0

    # What is the left-most pixel that can be involved in the computation?
    left = np.floor(u - kernel_width/2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = np.ceil(kernel_width) + 1

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left + np.arange(0,P,1) # = left + [0:1:P-1]

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    weights = scale * cubic(scale *(u-indices))
    weights = np.reshape(weights, [1,weights.size])
    return np.matmul(weights.T,weights)


def matlab_style_gauss2D(shape=(7,7),sigma=1.6):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x