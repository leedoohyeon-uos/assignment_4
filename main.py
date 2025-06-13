# main.py
"""
α,β-CROWN 외부 모델 실행 통합 스크립트
이 스크립트는 모델 생성, 변환, 스펙 생성, 검증 실행을 모두 통합합니다.
"""

import os
import sys
import argparse
import subprocess
import yaml
import torch
import torch.nn as nn
import torch.onnx
import torchvision
import torchvision.transforms as transforms
import numpy as np
import re
from pathlib import Path

# 모델 정의 클래스들
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModelTrainer:
    """모델 훈련 및 저장을 담당하는 클래스"""
    
    def __init__(self, model_type='cifar10'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_cifar10_model(self, epochs=5, save_path='cifar10_cnn.pth'):
        """CIFAR-10 모델 훈련"""
        print(f"CIFAR-10 모델 훈련 시작 (에포크: {epochs})")
        
        # 데이터 로드
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                shuffle=True, num_workers=2)
        
        # 모델 초기화
        model = CIFAR10CNN().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 훈련
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if i % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {i}, Loss: {loss.item():.4f}')
        
        # 모델 저장
        torch.save(model.state_dict(), save_path)
        print(f"모델이 {save_path}에 저장되었습니다.")
        return model
    
    def train_mnist_model(self, epochs=5, save_path='mnist_net.pth'):
        """MNIST 모델 훈련"""
        print(f"MNIST 모델 훈련 시작 (에포크: {epochs})")
        
        # 데이터 로드
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                shuffle=True, num_workers=2)
        
        # 모델 초기화
        model = MNISTNet().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 훈련
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if i % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {i}, Loss: {loss.item():.4f}')
        
        # 모델 저장
        torch.save(model.state_dict(), save_path)
        print(f"모델이 {save_path}에 저장되었습니다.")
        return model

class ModelConverter:
    """모델을 ONNX 형식으로 변환하는 클래스"""
    
    @staticmethod
    def convert_to_onnx(model, input_shape, onnx_path, model_type='cifar10'):
        """PyTorch 모델을 ONNX로 변환"""
        print(f"모델을 ONNX 형식으로 변환 중: {onnx_path}")
        
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                             'output': {0: 'batch_size'}}
            )
            print(f"ONNX 변환 완료: {onnx_path}")
            return True
        except Exception as e:
            print(f"ONNX 변환 실패: {e}")
            return False

class SpecGenerator:
    """VNNLIB 스펙 파일 생성 클래스"""
    
    @staticmethod
    def create_vnnlib_spec(input_size, output_size, epsilon=0.01, 
                          true_label=0, output_file="spec.vnnlib"):
        """VNNLIB 스펙 파일 생성"""
        print(f"VNNLIB 스펙 파일 생성 중: {output_file}")
        
        try:
            with open(output_file, 'w') as f:
                # 변수 선언
                f.write(f"; Robustness specification\n")
                f.write(f"; Input variables\n")
                for i in range(input_size):
                    f.write(f"(declare-const X_{i} Real)\n")
                
                f.write(f"; Output variables\n")
                for i in range(output_size):
                    f.write(f"(declare-const Y_{i} Real)\n")
                
                # 입력 제약 조건
                f.write(f"\n; Input constraints (L-infinity bound)\n")
                base_image = np.random.rand(input_size) * 2 - 1
                
                for i in range(input_size):
                    lower_bound = max(-1.0, base_image[i] - epsilon)
                    upper_bound = min(1.0, base_image[i] + epsilon)
                    f.write(f"(assert (>= X_{i} {lower_bound}))\n")
                    f.write(f"(assert (<= X_{i} {upper_bound}))\n")
                
                # 출력 제약 조건
                f.write(f"\n; Output constraints (adversarial property)\n")
                for i in range(output_size):
                    if i != true_label:
                        f.write(f"(assert (>= Y_{true_label} Y_{i}))\n")
            
            print(f"VNNLIB 스펙 파일 생성 완료: {output_file}")
            return True
        except Exception as e:
            print(f"VNNLIB 스펙 파일 생성 실패: {e}")
            return False

class ConfigGenerator:
    """α,β-CROWN 설정 파일 생성 클래스"""
    
    @staticmethod
    def create_config(model_name, model_path, config_path="config.yaml", 
                     epsilon=0.01, timeout=300):
        """YAML 설정 파일 생성"""
        print(f"α,β-CROWN 설정 파일 생성 중: {config_path}")
        
        config = {
            'model': {
                'name': model_name,
                'path': model_path
            },
            'data': {
                'dataset': model_name,
                'mean': [0.5, 0.5, 0.5] if 'cifar' in model_name else [0.1307],
                'std': [0.5, 0.5, 0.5] if 'cifar' in model_name else [0.3081]
            },
            'specification': {
                'norm': 'inf',
                'epsilon': epsilon
            },
            'solver': {
                'batch_size': 64,
                'alpha-crown': {
                    'lr_alpha': 0.1,
                    'iteration': 100
                },
                'beta-crown': {
                    'lr_beta': 0.05,
                    'iteration': 50
                }
            },
            'bab': {
                'timeout': timeout,
                'get_upper_bound': True
            },
            'general': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'complete_verifier': 'bab',
                'enable_incomplete_verification': True
            }
        }
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"설정 파일 생성 완료: {config_path}")
            return True
        except Exception as e:
            print(f"설정 파일 생성 실패: {e}")
            return False

class ABCrownRunner:
    """α,β-CROWN 검증 실행 클래스"""
    
    def __init__(self, abcrown_path="alpha-beta-CROWN/complete_verifier"):
        self.abcrown_path = abcrown_path
    
    def run_verification(self, model_path, spec_path, config_path, 
                        result_file="verification_results.txt"):
        """α,β-CROWN 검증 실행"""
        print("α,β-CROWN 검증 시작...")
        
        if not os.path.exists(self.abcrown_path):
            print(f"α,β-CROWN 경로를 찾을 수 없습니다: {self.abcrown_path}")
            return False
        
        # 절대 경로로 변환
        model_path = os.path.abspath(model_path)
        spec_path = os.path.abspath(spec_path)
        config_path = os.path.abspath(config_path)
        
        # 검증 명령어 구성
        cmd = [
            sys.executable, "abcrown.py",
            "--config", config_path,
            "--model", model_path,
            "--spec", spec_path,
            "--verbose"
        ]
        
        try:
            # 실행
            result = subprocess.run(cmd, 
                                   cwd=self.abcrown_path,
                                   capture_output=True, 
                                   text=True,
                                   timeout=600)
            
            # 결과 저장
            with open(result_file, 'w') as f:
                f.write("=== α,β-CROWN 검증 결과 ===\n")
                f.write(f"명령어: {' '.join(cmd)}\n")
                f.write(f"반환 코드: {result.returncode}\n\n")
                f.write("=== 표준 출력 ===\n")
                f.write(result.stdout)
                f.write("\n=== 표준 에러 ===\n")
                f.write(result.stderr)
            
            print(f"검증 완료. 결과가 {result_file}에 저장되었습니다.")
            
            # 결과 출력
            print("=== 검증 결과 ===")
            print(result.stdout)
            
            if result.stderr:
                print("=== 에러 메시지 ===")
                print(result.stderr)
                
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("검증 시간이 초과되었습니다.")
            return False
        except Exception as e:
            print(f"검증 실행 중 오류가 발생했습니다: {e}")
            return False

class ResultAnalyzer:
    """검증 결과 분석 클래스"""
    
    @staticmethod
    def parse_results(log_file):
        """α,β-CROWN 결과 파싱"""
        print(f"결과 분석 중: {log_file}")
        
        results = {
            'verified': False,
            'timeout': False,
            'unknown': False,
            'time_taken': None,
            'lower_bound': None,
            'upper_bound': None
        }
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # 검증 결과 추출
            if 'VERIFIED' in content:
                results['verified'] = True
            elif 'TIMEOUT' in content:
                results['timeout'] = True
            elif 'UNKNOWN' in content:
                results['unknown'] = True
                
            # 시간 정보 추출
            time_match = re.search(r'Total time: (\d+\.\d+)', content)
            if time_match:
                results['time_taken'] = float(time_match.group(1))
                
            # 바운드 정보 추출
            lower_match = re.search(r'Lower bound: (-?\d+\.\d+)', content)
            if lower_match:
                results['lower_bound'] = float(lower_match.group(1))
                
            upper_match = re.search(r'Upper bound: (-?\d+\.\d+)', content)
            if upper_match:
                results['upper_bound'] = float(upper_match.group(1))
                
            print("결과 분석 완료:")
            for key, value in results.items():
                print(f"  {key}: {value}")
                
        except FileNotFoundError:
            print(f"결과 파일 {log_file}을 찾을 수 없습니다.")
            
        return results

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='α,β-CROWN 외부 모델 검증 통합 실행')
    parser.add_argument('--model-type', choices=['cifar10', 'mnist'], 
                       default='cifar10', help='모델 타입')
    parser.add_argument('--epochs', type=int, default=5, 
                       help='훈련 에포크 수')
    parser.add_argument('--epsilon', type=float, default=0.01, 
                       help='검증 epsilon 값')
    parser.add_argument('--timeout', type=int, default=300, 
                       help='검증 타임아웃 (초)')
    parser.add_argument('--skip-training', action='store_true', 
                       help='훈련 건너뛰기 (기존 모델 사용)')
    parser.add_argument('--abcrown-path', default='alpha-beta-CROWN/complete_verifier',
                       help='α,β-CROWN 경로')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("α,β-CROWN 외부 모델 검증 시작")
    print("=" * 50)
    
    # 파일 경로 설정
    model_name = args.model_type
    pth_path = f"{model_name}_model.pth"
    onnx_path = f"{model_name}_model.onnx"
    spec_path = f"{model_name}_spec.vnnlib"
    config_path = f"{model_name}_config.yaml"
    result_path = f"{model_name}_results.txt"
    
    # 1. 모델 훈련 (선택사항)
    if not args.skip_training:
        print("\n1. 모델 훈련 단계")
        trainer = ModelTrainer(args.model_type)
        
        if args.model_type == 'cifar10':
            model = trainer.train_cifar10_model(args.epochs, pth_path)
            input_shape = (3, 32, 32)
            input_size = 3072
        elif args.model_type == 'mnist':
            model = trainer.train_mnist_model(args.epochs, pth_path)
            input_shape = (1, 28, 28)
            input_size = 784
        
        output_size = 10
    else:
        print("\n1. 모델 훈련 건너뛰기")
        
        # 기존 모델 로드
        if args.model_type == 'cifar10':
            model = CIFAR10CNN()
            input_shape = (3, 32, 32)
            input_size = 3072
        elif args.model_type == 'mnist':
            model = MNISTNet()
            input_shape = (1, 28, 28)
            input_size = 784
            
        if os.path.exists(pth_path):
            model.load_state_dict(torch.load(pth_path))
            print(f"기존 모델 로드: {pth_path}")
        else:
            print(f"기존 모델을 찾을 수 없습니다: {pth_path}")
            print("--skip-training 옵션 없이 다시 실행하세요.")
            return
        
        output_size = 10
    
    # 2. ONNX 변환
    print("\n2. ONNX 변환 단계")
    if not ModelConverter.convert_to_onnx(model, input_shape, onnx_path, args.model_type):
        print("ONNX 변환 실패")
        return
    
    # 3. VNNLIB 스펙 생성
    print("\n3. VNNLIB 스펙 생성 단계")
    if not SpecGenerator.create_vnnlib_spec(input_size, output_size, 
                                          args.epsilon, 0, spec_path):
        print("VNNLIB 스펙 생성 실패")
        return
    
    # 4. 설정 파일 생성
    print("\n4. 설정 파일 생성 단계")
    if not ConfigGenerator.create_config(model_name, onnx_path, config_path, 
                                       args.epsilon, args.timeout):
        print("설정 파일 생성 실패")
        return
    
    # 5. α,β-CROWN 검증 실행
    print("\n5. α,β-CROWN 검증 실행 단계")
    runner = ABCrownRunner(args.abcrown_path)
    success = runner.run_verification(onnx_path, spec_path, config_path, result_path)
    
    # 6. 결과 분석
    print("\n6. 결과 분석 단계")
    results = ResultAnalyzer.parse_results(result_path)
    
    # 최종 결과 출력
    print("\n" + "=" * 50)
    print("최종 결과")
    print("=" * 50)
    
    if results['verified']:
        print("✅ 모델이 검증되었습니다 (안전함)")
    elif results['timeout']:
        print("⏰ 검증 시간이 초과되었습니다")
    elif results['unknown']:
        print("❓ 검증 결과를 확인할 수 없습니다")
    else:
        print("❌ 모델이 검증되지 않았습니다 (안전하지 않음)")
    
    if results['time_taken']:
        print(f"검증 시간: {results['time_taken']:.2f}초")
    
    print(f"\n생성된 파일들:")
    print(f"  - 모델 파일: {pth_path}, {onnx_path}")
    print(f"  - 스펙 파일: {spec_path}")
    print(f"  - 설정 파일: {config_path}")
    print(f"  - 결과 파일: {result_path}")
    
    print("\n검증 완료!")

if __name__ == "__main__":
    main()