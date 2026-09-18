#!/usr/bin/python

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np

def dct(x, norm='ortho', axis=0):
    """
    Orthonormal DCT-II via a direct O(N^2) matrix build (no scipy).

    Parameters:
        x: `input array`
        norm: `only 'ortho' supported (matches scipy.fftpack norm='ortho')`
        axis: `axis along which to transform`
    """
    if norm != 'ortho':
        raise NotImplementedError("only norm='ortho' supported")
    return _dct_apply(x, axis, _dct2_matrix(x.shape[axis]))

def idct(x, norm='ortho', axis=0):
    """
    Orthonormal DCT-III (inverse of the orthonormal DCT-II), O(N^2) matrix build.

    Parameters:
        x: `input array`
        norm: `only 'ortho' supported (matches scipy.fftpack norm='ortho')`
        axis: `axis along which to transform`
    """
    if norm != 'ortho':
        raise NotImplementedError("only norm='ortho' supported")
    return _dct_apply(x, axis, _dct3_matrix(x.shape[axis]))

def _dct2_matrix(N):
    # orthonormal DCT-II basis: X_k = s_k * sum_n x_n cos(pi*(2n+1)*k/(2N))
    n = np.arange(N).reshape(1, N)               # sample index
    k = np.arange(N).reshape(N, 1)               # coefficient index
    C = np.cos(np.pi * (2*n + 1) * k / (2*N))
    C *= np.sqrt(2.0 / N)                        # scale all rows
    C[0] *= (1.0 / np.sqrt(2))                   # k = 0 row is unit norm
    return C

def _dct3_matrix(N):
    # inverse (transpose) of the orthonormal DCT-II basis
    return _dct2_matrix(N).T

def _dct_apply(x, axis, T):
    return np.moveaxis(np.tensordot(T, np.moveaxis(x, axis, 0), axes=1), 0, axis)

def gen_test_signal(k=20, n=1000, amps_l=-100, amps_h=100, snr_db=None):
    (y, A, k, x_t, x_f) = gen_rand_gauss_signal(k, n, amps_l, amps_h, snr_db)
    return (y, A, x_t, x_f)

def add_awgn(x_volts, target_snr_db):
    # calculate signal power and convert to dB
    sig_avg_watts = np.mean(x_volts ** 2)
    sig_avg_db = 10 * np.log10(sig_avg_watts)

    # calculate noise according to SNR_dB = P_signal,dB - P_noise,dB then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)

    # generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), x_volts.shape)

    # noise up the original signal
    y_volts = x_volts + noise_volts

    return y_volts

def gen_rand_gauss_signal(k, n, amps_l, amps_h, snr_db):
    # number of samples needed
    c = 2
    m = int(np.ceil(c * k * np.log(n/k)))
    A = ((1.0/m)**0.5) * np.random.randn(m, n)

    # generate signal
    signal_f = np.zeros((n, 1))
    pos = np.random.randint(0, n, (k, 1))
    for i in range(0, len(pos)):
        signal_f[pos[i]] = np.random.randint(amps_l, amps_h)
    signal_t = idct(signal_f, norm='ortho', axis=0)

    # add noise
    if snr_db:
        signal_t = add_awgn(signal_t, snr_db)
        signal_f = dct(signal_t, norm='ortho', axis=0)

    # sample the signal
    y = np.dot(A, signal_t)

    D = dct(np.eye(A.shape[1]), norm='ortho', axis=0)
    Ap = np.dot(A, D.T)

    return (y, Ap, k, signal_t, signal_f)

def plt_error(x_hat, x_f, title, alg=None):
    # compute the error
    sse = np.sum((np.ravel(x_hat) - np.ravel(x_f))**2)
    plt.figure(0)
    plt.stem(x_hat,  markerfmt='ro')
    plt.stem(x_f,  markerfmt='b-')
    plt.title(title + f', sse = {sse}')
    plt.savefig(outname(alg or title), dpi=110)
    plt.close()

def outname(alg, tag=''):
    """
    collision-free output name: folder/algo_[tag_]YYYYmmdd_HHMMSS

    Every figure carries the generating algorithm prefix and a
    timestamp so repeated runs never overwrite earlier results.
    """
    import datetime
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base = slug(str(alg))
    return f'out/{base}{("_" + slug(tag)) if tag else ""}_{stamp}.png'

def slug(text):
    """
    filesystem-safe title fragment
    """
    keep = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    slug = ''.join(c if c in keep else '_' for c in str(text))
    return slug[:80]

def plt_signal(signal, title, alg=None):
    plt.plot(signal)
    plt.title(title)
    plt.grid()
    plt.savefig(outname(alg or title), dpi=110)
    plt.close()

if __name__ == "__main__":
    # standalone: visualize the standard signal pair
    (y, A, x_t, x_f) = gen_test_signal(snr_db=None, k=10, n=200, amps_l=-10, amps_h=10)
    plt_signal(x_t, 'Time domain signal_t', alg='signal_time')
    plt_signal(x_f, 'Frequency domain signal_f', alg='signal_freq')
