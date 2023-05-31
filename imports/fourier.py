import numpy as np

class Fourier:
    def __init__(self, signal):
        self.signal = signal
        

    def compute_fourier_series(self, x, y, num_harmonics):
        n = len(x)
        t = np.linspace(min(x), max(x), num=n)
        y_fourier = np.zeros_like(t)

        for i in range(1, num_harmonics + 1):
            cos_component = np.cos(i * 2 * np.pi * t / max(x))
            sin_component = np.sin(i * 2 * np.pi * t / max(x))

            a_i = 2 / n * np.sum(y * cos_component)
            b_i = 2 / n * np.sum(y * sin_component)

            y_fourier += a_i * cos_component + b_i * sin_component

        return t, y_fourier

    def compute_fourier_transform(self, x, y, zero_padding=False):
        n = len(x)
        freq = np.fft.fftfreq(n, d=x[1] - x[0])
        y_fourier = np.fft.fft(y, n=zero_padding*n if zero_padding else n)

        return freq, y_fourier

    def analyze_fourier_coefficients(self, freq, y_fourier, num_coefficients):
        sorted_indices = np.argsort(np.abs(y_fourier))[::-1]
        top_indices = sorted_indices[:num_coefficients]

        analyzed_freq = freq[top_indices]
        analyzed_coefficients = y_fourier[top_indices]

        return analyzed_freq, analyzed_coefficients

    def compute_power_spectrum(self, freq, y_fourier):
        power_spectrum = np.abs(y_fourier) ** 2
        return freq, power_spectrum

    def compute_autocorrelation(self, x, y):
        y_mean = np.mean(y)
        y_normalized = y - y_mean
        autocorr = np.correlate(y_normalized, y_normalized, mode='full')
        autocorr /= np.max(autocorr)
        lag = np.arange(-len(y) + 1, len(y))

        return lag, autocorr
