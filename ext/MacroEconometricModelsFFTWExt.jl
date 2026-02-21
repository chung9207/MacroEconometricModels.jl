module MacroEconometricModelsFFTWExt

using MacroEconometricModels
using FFTW

function __init__()
    MacroEconometricModels._FFT_IMPL[] = FFTW.fft
    MacroEconometricModels._IFFT_IMPL[] = FFTW.ifft
end

end
