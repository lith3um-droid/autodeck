"""Stand-ins for the three essentia algorithms autodeck uses, for platforms without essentia.

essentia ships wheels for macOS and manylinux only, so on Windows these numpy/librosa ports take
over (the call sites try `import essentia.standard` first and fall back here). Each module is a
port of the essentia C++ it is named after and was checked against the real thing on the same
signals (see tests/test_compat.py):

    musicnn.py       TensorflowInputMusiCNN: the mel front-end of the Discogs-EffNet style model
    key.py           KeyExtractor (FrameCutter > Windowing > Spectrum > SpectralPeaks >
                     SpectralWhitening > HPCP > Key, profile "edma")
    danceability.py  Danceability (detrended fluctuation analysis)
"""
