import librosa


def load_audio(filepath, start_time=0.0, end_time=None, samp_freq=16000):
    """
    Load audio from disk.
    """
    duration = end_time - start_time
    assert (
        duration > 0
    ), "Duration is negative or zero, check that end_time > start_time"
    audio, sr = librosa.load(
        filepath, sr=samp_freq, offset=start_time, duration=duration
    )
    return audio
