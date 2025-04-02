import numpy as np
import os

# 확인할 .npy 파일 경로
npy_path = "/Users/simjuheun/Desktop/myProject/Hifi-Gan_Legacy/dataSource/sourceAudio.npy"

# 파일 존재 확인
if not os.path.isfile(npy_path):
    print(f"❌ 파일이 존재하지 않습니다: {npy_path}")
    exit(1)

# 로드
mel = np.load(npy_path)

# 구조 출력
print(f"📂 파일 로드 완료: {npy_path}")
print(f"📐 Shape: {mel.shape}  (n_mels x T)")
print("📊 값 통계:")
print(f"   Min:  {mel.min():.6f}")
print(f"   Max:  {mel.max():.6f}")
print(f"   Mean: {mel.mean():.6f}")
print(f"   Std:  {mel.std():.6f}")