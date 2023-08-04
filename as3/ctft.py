import matplotlib.pyplot as plt
import numpy as np


def ctft(ss, tt, ff, plot=False):

    m = np.exp(-1j*np.pi*np.outer(ff, tt))  # Fourier Matrix
    x = (tt[1] - tt[0])*m@ss[:, None]       # Fourier Transform
    absx = np.abs(x)                        # Magnitude
    angx = np.unwrap(np.angle(x))           # Unwrapped Phase

    if not plot:
        return x, absx, angx
    
    fig, axes = plt.subplots(1, 3)
    fig.set_figheight(5)
    fig.set_figwidth(20)

    # Signal in time:
    axes[0].plot(ss, linewidth=1)
    axes[0].set_xlim([0, len(ss)])
    axes[0].set_ylim([min(ss), max(ss)])
    axes[0].set_xlabel("Time $(s)$")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Signal in time")
    axes[0].grid()

    # Fourier Magnitude:
    axes[1].plot(ff, absx, linewidth=2)
    axes[1].set_xlim([ff[0], ff[-1]+1])
    axes[1].set_ylim([min(absx), max(absx)])
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("Magnitude of CT FT")
    axes[1].grid()

    # Unwrapped Fourier Phase:
    axes[2].plot(ff, angx, linewidth=2)
    axes[2].set_xlim([ff[0], ff[-1]+1])
    axes[2].set_ylim([min(angx), max(angx)])
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Radians")
    axes[2].set_title("Phase of CT FT")
    axes[2].grid()

    plt.draw()

    return x, absx, angx

