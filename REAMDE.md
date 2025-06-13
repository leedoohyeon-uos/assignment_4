# α,β-CROWN 외부 모델 검증 - 빠른 시작 가이드

### 1. 환경 준비
```bash
# 저장소 클론
git clone <repository-url>
cd abcrown-external-verification

# Docker 및 NVIDIA Container Toolkit 설치 확인
docker --version
nvidia-docker --version  # 또는 docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### 2. 즉시 실행
```bash
# GPU 지원 환경에서 실행
docker-compose up -d
docker exec -it abcrown-verification /workspace/run_examples.sh

# CPU 전용 실행
docker run -it --rm \
  -v $(pwd):/workspace \
  abcrown-external:latest \
  python main.py --model-type mnist
```

### 3. 예상 결과
```
================================== 
α,β-CROWN 외부 모델 검증 시작
==================================
MNIST 모델: VERIFIED (45-80초)
CIFAR-10 모델: UNKNOWN (180-300초)
```

### 환경 설정
```bash
# Python 3.9+ 환경
conda create -n abcrown python=3.9
conda activate abcrown

# 의존성 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# α,β-CROWN 설치
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN && pip install -r requirements.txt
cd ..
```

### 실행
```bash
# 기본 실행
python main.py --model-type mnist

# 고급 옵션
python main.py --model-type cifar10 --epochs 10 --epsilon 0.005 --timeout 600
```

## 주요 실행 옵션

| 옵션 | 설명 | 기본값 | 예시 |
|------|------|--------|------|
| `--model-type` | 모델 선택 | `cifar10` | `mnist`, `cifar10` |
| `--epochs` | 훈련 에포크 | `5` | `10` |
| `--epsilon` | 섭동 크기 | `0.01` | `0.005` |
| `--timeout` | 검증 제한시간 | `300` | `600` |
| `--skip-training` | 기존 모델 사용 | `False` | - |

## 결과 해석

### 성공적인 검증 (VERIFIED)
```
모델이 검증되었습니다 (안전함)
검증 시간: 67.34초
Lower bound: 0.0234
Upper bound: 0.1876
```
→ **해석**: 지정된 ε 범위 내에서 adversarial attack에 안전함

### 시간 초과 (TIMEOUT)
```
검증 시간이 초과되었습니다
Lower bound: -0.1234
Upper bound: 0.0876
```
→ **해석**: 완전 검증 불가, bound 정보로 부분적 안전성 분석 가능

### 알 수 없음 (UNKNOWN)
```
검증 결과를 확인할 수 없습니다
```
→ **해석**: 모델 복잡도나 설정 문제로 검증 실패

### 성능 최적화
```bash
# GPU 메모리 최적화
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# CPU 멀티스레딩
export OMP_NUM_THREADS=4

# 더 빠른 검증을 위한 설정
python main.py --model-type mnist --epsilon 0.005 --timeout 120
```

## 생성되는 파일들

```
프로젝트/
├── mnist_model.pth          # 훈련된 PyTorch 모델
├── mnist_model.onnx         # ONNX 변환 모델
├── mnist_spec.vnnlib        # 검증 스펙 파일
├── mnist_config.yaml        # α,β-CROWN 설정
├── mnist_results.txt        # 검증 결과 로그
└── data/                    # 데이터셋 캐시
```

## 실전 활용 예시

### 1. 빠른 프로토타입 검증
```bash
# 5분 내 결과 확인
docker run --rm abcrown-external python main.py --model-type mnist --epochs 2
```

### 2. 정밀 안전성 분석
```bash
# 엄격한 검증 (더 작은 ε)
python main.py --model-type mnist --epsilon 0.003 --timeout 900
```

### 3. 배치 실험
```bash
# 여러 설정으로 자동 실험
for eps in 0.005 0.01 0.02; do
  python main.py --model-type mnist --epsilon $eps --timeout 300
done
```