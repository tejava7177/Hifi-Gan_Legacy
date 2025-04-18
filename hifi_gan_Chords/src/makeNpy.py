import os
import numpy as np
import librosa

# ✅ 입력 경로
wav_path = "/dataSource/sourceAudio.wav"

# ✅ 출력 경로
output_dir = "/dataSource"
file_id = os.path.splitext(os.path.basename(wav_path))[0]
output_npy = os.path.join(output_dir, f"{file_id}.npy")

# ✅ 디렉토리 보장
os.makedirs(output_dir, exist_ok=True)

# ✅ MEL 파라미터 (Universal HiFi-GAN 기준)
sr = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 80
fmin = 0
fmax = 8000

# ✅ WAV 로드
y, _ = librosa.load(wav_path, sr=sr)

# ✅ MEL 추출 (power scale, dB 변환 없음)
mel_spec = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    n_mels=n_mels,
    fmin=fmin,
    fmax=fmax
)

# ✅ 저장
np.save(output_npy, mel_spec)
print(f"✅ MEL 저장 완료: {output_npy}")
print(f"🔍 MEL shape: {mel_spec.shape} | min: {mel_spec.min():.5f}, max: {mel_spec.max():.5f}, mean: {mel_spec.mean():.5f}")