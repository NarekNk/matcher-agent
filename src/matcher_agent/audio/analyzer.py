from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import essentia.standard as es
except Exception:  # pragma: no cover - optional dependency for tests
    es = None


def analyze_audio(audio_path: Path) -> dict | None:
    if es is None:
        return None
    try:
        loader = es.MonoLoader(filename=str(audio_path), sampleRate=44100)
        audio = loader()
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _, beats_confidence, _, _ = rhythm_extractor(audio)
        loudness = es.Loudness()(audio)
        w = es.Windowing(type="hann")
        spectrum = es.Spectrum()
        centroid = es.SpectralCentroidTime()
        rolloff = es.RollOff()
        flux = es.Flux()
        zcr = es.ZeroCrossingRate()
        mfcc_algo = es.MFCC(numberCoefficients=13)

        centroids, rolloffs, fluxes, zcrs = [], [], [], []
        mfccs = []
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024, startFromZero=True):
            win = w(frame)
            spec = spectrum(win)
            centroids.append(centroid(frame))
            rolloffs.append(rolloff(spec))
            fluxes.append(flux(spec))
            zcrs.append(zcr(frame))
            _, mfcc_frame = mfcc_algo(spec)
            mfccs.append(mfcc_frame)
        if not mfccs:
            return None
        mfccs_arr = np.array(mfccs, dtype=np.float64)

        danceability, _ = es.Danceability()(audio)
        energy = es.Energy()(audio)

        features = {
            "bpm": bpm,
            "beats_confidence": beats_confidence,
            "loudness": float(loudness),
            "danceability": float(danceability),
            "energy": float(energy),
            "spectral_centroid": float(np.mean(centroids)),
            "spectral_rolloff": float(np.mean(rolloffs)),
            "spectral_flux": float(np.mean(fluxes)),
            "zcr": float(np.mean(zcrs)),
        }
        for i, val in enumerate(mfccs_arr.mean(axis=0)):
            features[f"mfcc_{i+1}"] = float(val)
        return features
    except Exception:
        return None
