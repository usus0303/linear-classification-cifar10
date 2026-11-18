# CIFAR-10 Linear Classification Assignment

**2026-11-18 | Linear Classification | No Training**

## 과제 소개

동국대학교 컴퓨터 비전 수업의 **Week 9-2 (Linear Classification)** 강의 PPT를 기준으로 구현했습니다.
CIFAR-10 이미지를 사용해 무작위로 초기화한 선형 분류기 `f(x) = Wx + b`의 동작을 확인합니다.

**가중치 업데이트, 모델 학습, 정확도/정밀도/재현율/F1 계산은 진행하지 않습니다.**

### 구현 내용

1. `W`: 크기 **10×3072**, `N(0, 0.01²)` 무작위 초기화 (`seed=42`)
2. `b`: 크기 **10**, 모든 원소 0
3. `x`: CIFAR-10 이미지(32×32×3)를 길이 3072 벡터로 사용, 값은 255로 나눔
4. 점수: `scores = X @ W.T + b` (각 이미지에서 `f(x) = Wx + b`와 같음)
5. 예측: 최고 점수 클래스 `argmax`
6. SVM Hinge Loss: `L_i = Σ_(j≠y_i) max(0, s_j − s_y + 1)`
7. Softmax: `P_j = exp(s_j − max(s)) / Σ_k exp(s_k − max(s))`
8. 올바른/잘못된 분류 이미지 및 각 클래스의 32×32×3 가중치 시각화

## 1. 설치 방법 (VS Code / Windows 터미널)

Python 3.10 이상을 권장합니다. **GPU, PyTorch, sklearn이 필요하지 않습니다.**

```powershell
pip install -r requirements.txt
```

## 2. CIFAR-10 준비

아래 사이트에서 **CIFAR-10 python version**을 다운로드하세요.

- 공식 안내: https://cave.cs.toronto.edu/kriz/cifar.html
- Python 압축 파일: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

다운로드한 `cifar-10-python.tar.gz`를 `linear_classification.py`와 **같은 폴더**에 놓습니다.
압축을 풀 필요는 없습니다. KNN 과제에서 사용한 파일을 그대로 재사용할 수 있습니다.

또는 명령에 `--download`를 붙이면 원본 파일이 없을 때 자동 다운로드합니다.

## 3. 실행 방법

### 이미 파일이 있는 경우

```powershell
python linear_classification.py --data cifar-10-python.tar.gz
```

### 파일이 없어서 자동으로 다운로드해야 하는 경우

```powershell
python linear_classification.py --download
```

### 예시 이미지 20장으로 변경

```powershell
python linear_classification.py --data cifar-10-python.tar.gz --num-examples 20
```

### 수식만 검증 (데이터셋 없어도 가능)

```powershell
python linear_classification.py --self-test
python -m unittest -v test_linear_classification.py
```

### 데이터가 다른 폴더에 있는 경우

```powershell
python linear_classification.py --data "C:\Users\사용자\Downloads\cifar-10-python.tar.gz"
```

압축을 푼 폴더를 이용하는 방법도 지원합니다.

```powershell
python linear_classification.py --data "C:\Users\사용자\Downloads\cifar-10-batches-py"
```

## 4. 결과 파일

실제 CIFAR-10을 준비하고 메인 스크립트를 실행하면 다음 파일들이 `results/`에 저장됩니다.

| 파일 | 과제 관련 내용 |
|---|---|
| `score_probabilities.csv` | 모든 선택 이미지의 **10개 클래스별 점수 및 Softmax 확률** |
| `svm_loss_and_probability_sums.csv` | 각 이미지 SVM Loss 및 확률 합 |
| `brief_explanations_ko.md` | 실제 평균 SVM Loss와 결과에 대한 짧은 설명 |
| `selected_examples.png` | 선택된 이미지의 정답 / 예측 비교 |
| `correct_incorrect.png` | 정답 사례 / 오답 사례 |
| `score_vs_softmax.png` | 같은 이미지의 점수와 Softmax 확률 비교 |
| `random_weights.png` | 무작위 가중치를 32×32×3 이미지로 시각화 |

예시를 기본값인 **10장**으로 설정하면, 결과 CSV는 10장 × 10개 클래스 = **100행**으로 만들어집니다.

> 참고: 제출용 결과는 사용자 PC에서 **공식 CIFAR-10 데이터셋으로 실행한 다음 생성된 결과 파일**을 올려야 합니다. `results/ppt_formula_self_test.txt`는 PPT 예제 검증 결과이며 CIFAR-10 예측 결과가 아닙니다.

## 5. 설명

**선형 점수와 예측**: CIFAR-10의 각 이미지는 32×32×3=3072개 픽셀을 가집니다. 이 값을 벡터로 변환한 후 `W(10×3072)`와 곱하고 `b(10)`을 더하면 10개의 클래스 점수가 나옵니다. 분류기는 가장 높은 점수를 갖는 클래스를 선택합니다.

**SVM Hinge Loss**: 정답 클래스 점수보다 다른 클래스 점수가 더 높거나, 정답 점수가 다른 클래스 점수보다 1 이상 높지 않을 때 손실이 발생합니다. 각 이미지의 손실을 합한 뒤 선택 이미지 수로 나누어 평균을 구합니다.

**Softmax**: 원본 점수는 음수도 가능하며 확률이 아닙니다. Softmax는 모든 클래스 값을 0 이상으로 만들고 총합을 1로 정규화합니다. 같은 수를 모든 점수에서 빼도 결과는 달라지지 않으므로, 계산 안정성을 위해 가장 큰 값을 먼저 뺍니다. Softmax를 사용해도 최고 점수 클래스는 바뀌지 않습니다.

**정답/오답**: 모델이 훈련되지 않아 가중치는 이미지와 클래스 사이의 특징을 반영하지 못합니다. 따라서 틀린 예가 많을 수 있으며, 맞는 예도 우연에 의한 것으로 해석해야 합니다.

**무작위 가중치 이미지**: 각 클래스의 W 벡터를 32×32×3 형태로 복원하면 잡음 형태의 이미지가 나타납니다. 이는 학습을 통해 특정 클래스의 시각적 패턴을 반영하도록 조정된 가중치가 아니기 때문입니다.

## 6. GitHub 제출 예시

`results/`에 실제 실행 결과가 생성됐는지 확인한 후 아래 명령어를 실행합니다.

```powershell
git init
git add .
git commit -m "Add CIFAR-10 random linear classifier assignment"
git branch -M main
git remote add origin https://github.com/<your-username>/<new-repository>.git
git push -u origin main
```

`<your-username>/<new-repository>` 부분은 새로 만든 GitHub 저장소 주소로 교체하세요.

**주의:** `cifar-10-python.tar.gz`는 크기가 크므로 `.gitignore`에 포함해 GitHub에 업로드하지 않습니다. 과제에서는 소스와 결과 파일만 업로드하면 됩니다.
