from __future__ import annotations

from pathlib import Path

import numpy as np

_ES_IMPORT_ERROR: Exception | None = None
try:
    import essentia.standard as es
except Exception as exc:  # pragma: no cover - optional dependency for tests
    es = None
    _ES_IMPORT_ERROR = exc


def analyze_audio(audio_path: Path) -> dict | None:
    """Extract audio features from a local audio file using Essentia.

    Returns a flat dict of feature names to float values, or None on
    failure.  Features include:

    * **Rhythm**: bpm, beats_confidence, onset_rate
    * **Global**: loudness, danceability, energy
    * **Key**: key (0-11 semitones from C), mode (0=minor, 1=major)
    * **Spectral means**: spectral_centroid, spectral_rolloff,
      spectral_flux, zcr
    * **Spectral stds** (temporal variation): spectral_centroid_std,
      spectral_rolloff_std, spectral_flux_std, zcr_std
    * **MFCCs**: mfcc_1 … mfcc_13 (frame means)
    * **MFCC stds**: mfcc_1_std … mfcc_13_std (frame std devs)
    """
    if es is None:
        print(
            "Audio analysis skipped: essentia is not available (%s)",
            repr(_ES_IMPORT_ERROR) if _ES_IMPORT_ERROR else "import failed",
        )
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
            print(
                "Audio analysis produced no MFCC frames (empty or too short audio): %s",
                audio_path,
            )
            return None
        mfccs_arr = np.array(mfccs, dtype=np.float64)

        danceability, _ = es.Danceability()(audio)
        energy = es.Energy()(audio)

        # --- Key & mode ---
        key_extractor = es.KeyExtractor()
        key_str, scale_str, key_strength = key_extractor(audio)
        key_int = _key_to_int(key_str)
        mode_int = 1 if scale_str.lower() == "major" else 0

        # --- Onset rate ---
        onset_rate_algo = es.OnsetRate()
        onsets, onset_rate = onset_rate_algo(audio)

        features: dict[str, float] = {
            "bpm": float(bpm),
            "beats_confidence": float(beats_confidence),
            "loudness": float(loudness),
            "danceability": float(danceability),
            "energy": float(energy),
            "key": float(key_int),
            "mode": float(mode_int),
            "onset_rate": float(onset_rate),
            "spectral_centroid": float(np.mean(centroids)),
            "spectral_rolloff": float(np.mean(rolloffs)),
            "spectral_flux": float(np.mean(fluxes)),
            "zcr": float(np.mean(zcrs)),
            # Temporal variation (frame-level std dev).
            "spectral_centroid_std": float(np.std(centroids)),
            "spectral_rolloff_std": float(np.std(rolloffs)),
            "spectral_flux_std": float(np.std(fluxes)),
            "zcr_std": float(np.std(zcrs)),
        }
        mfcc_means = mfccs_arr.mean(axis=0)
        mfcc_stds = mfccs_arr.std(axis=0)
        for i in range(mfccs_arr.shape[1]):
            features[f"mfcc_{i+1}"] = float(mfcc_means[i])
            features[f"mfcc_{i+1}_std"] = float(mfcc_stds[i])
        return features
    except Exception:
        print("Audio analysis failed for %s", audio_path)
        return None


_KEY_MAP: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def _key_to_int(key_str: str) -> int:
    """Convert a key string (e.g. 'C#') to a semitone integer 0-11."""
    return _KEY_MAP.get(key_str.strip(), -1)
