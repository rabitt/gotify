import librosa
import numpy as np

GOT_THEME = "got.wav"
TESTING_AUDIO = "barbiegirl.wav"
HOP = 2048

def load_got_theme():
    y, sr = librosa.load(GOT_THEME)
    stft = librosa.stft(y, hop_length=HOP, n_fft=HOP*4)
    cqt = librosa.cqt(y, hop_length=HOP)
    return stft, cqt, y, sr


def load_other_audio(audio_file):
    y, sr = librosa.load(audio_file)
    stft = librosa.stft(y, hop_length=HOP, n_fft=HOP*4)
    cqt = librosa.cqt(y, hop_length=HOP)
    return stft, cqt


def get_closest_indices(got_features, other_features):
    closest_indices = np.zeros((len(got_features.T), ))
    for i, frame in enumerate(got_features.T):
        diff = np.sum(np.abs(other_features.T - frame), axis=1)
        closest_indices[i] = np.argmin(diff)
    return closest_indices


def indices_to_stft(indices, old_stft, got_stft, use_got_phase=True):
    n_frames = len(indices)
    n_bins = len(old_stft)

    new_stft = np.zeros((n_bins, n_frames), dtype=old_stft.dtype)
    for i, idx in enumerate(indices):
        angles = np.angle(got_stft[:, i])
        magnitude = np.sum(np.abs(got_stft[:, i]))

        if use_got_phase:
            new_stft[:, i] = np.abs(old_stft[:, int(idx)])*np.exp(1j*angles)
        else:
            new_stft[:, i] = old_stft[:, int(idx)]

        orig_magnitude = np.sum(np.abs(new_stft[:, i]))
        if orig_magnitude == 0:
            orig_magnitude = 1.0
        new_stft[:, i] = magnitude*new_stft[:, i]/orig_magnitude
    return new_stft


def render_audio(new_stft, sr, fpath, y_orig, mix=False):
    assert not np.any(np.isnan(new_stft))
    audio = librosa.istft(new_stft, hop_length=HOP)
    if mix:
        min_len = np.min([len(audio), len(y_orig)])
        audio = audio[:min_len] + y_orig[:min_len]
    librosa.output.write_wav(fpath, audio, sr)


def main():
    print 'loading game of thrones theme...'
    got_stft, got_cqt, y, sr = load_got_theme()

    print 'loading other audio file...'
    other_stft, other_cqt = load_other_audio(TESTING_AUDIO)

    print 'finding best points...'
    closest_indices = get_closest_indices(got_cqt, other_cqt)
    print closest_indices

    print 'creating new stft...'
    new_stft = indices_to_stft(
        closest_indices, other_stft, got_stft, use_got_phase=False)

    print 'rendering audio'
    render_audio(new_stft, sr, TESTING_AUDIO[:-4] + "-got.wav", y)


if __name__ == main():
    main()
