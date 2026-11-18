"""CIFAR-10 선형 분류 실습: 연산 순서를 따라 읽는 한국어 주석판.

읽는 순서: 맨 위 main()에서 실행 흐름을 보고, 그 아래 함수들을 단계별로 읽기.
핵심 함수: preprocess_images → initialize_parameters → compute_scores →
          predict_classes → compute_svm_loss → compute_softmax.

실행 흐름:
    test_batch 불러오기: (10000,3072)
      → 클래스별 1장 선택: (10,3072)
      → X=픽셀/255: (10,3072)
      → W(10,3072), b(10,) 초기화: 이미지와 독립적으로 생성
      → scores=X@W.T+b: (10,10)
      → argmax로 예측: (10,)
      ├→ SVM: 정답 제외 나머지 9개 클래스와 비교, 이미지별 합 → 전체 평균
      └→ Softmax: 같은 원본 scores를 확률로 변환, 각 이미지의 10개 합은 1
      → 원본/예측 및 W를 시각화 (W 업데이트 없음)

대표 숫자(사용자가 실행한 seed=42, 10장 설정):
    첫 이미지 Test #885: true=airplane, pred=dog
    scores[0,0]=-0.353013, scores[0,5]=0.867669
    SVM 평균=8.909212, Softmax 확률의 합=1.

사용법: python linear_classification.py --data "cifar-10-batches-py"
        python linear_classification.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import pickle
import tarfile
import urllib.request
from pathlib import Path

import matplotlib

# VS Code 터미널 또는 화면이 없는 환경에서도 그래프를 파일로 저장할 수 있도록 설정합니다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# CIFAR-10 공식 클래스 순서입니다. 정답 숫자 0~9가 이 순서에 대응합니다.
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3
INPUT_DIM = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS  # 32 x 32 x 3 = 3,072
NUM_CLASSES = len(CLASS_NAMES)                            # 10개 클래스
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


# -----------------------------------------------------------------------------
# START HERE: main()의 1~10단계를 먼저 읽으면 데이터가 어떻게 계산되는지 보인다.
# 각 단계의 내부 연산은 아래에 순서대로 정의된 함수에서 확인한다.
# -----------------------------------------------------------------------------
def main() -> None:
    """전체 실험의 실제 실행 순서표. 발표 때는 이 함수부터 보면 된다.

    함수 호출의 흐름:
      데이터 불러오기 → 클래스별 1장 선택 → 0~1 정규화 → W,b 생성
        → Wx+b 점수 계산 → 최고 점수 예측
        ├─ SVM Loss 계산 (정답과 나머지 9개 클래스 비교)
        └─ Softmax 계산 (원본 점수 10개를 확률로 변환)
      → 정답/오답 비교 → 클래스별 W 패턴 시각화 → 결과 저장

    SVM Loss를 Softmax에 입력하지 않는다. 두 계산 모두 'scores'를 사용한다.
    과제 조건상 W는 절대로 업데이트하지 않는다.
    """
    parser = argparse.ArgumentParser(description="CIFAR-10 Linear Classification (no training)")
    parser.add_argument(
        "--data", type=Path, default=Path("cifar-10-python.tar.gz"),
        help="CIFAR-10 Python tar.gz 파일 또는 압축 해제된 폴더 경로",
    )
    parser.add_argument("--download", action="store_true", help="데이터가 없다면 공식 사이트에서 다운로드")
    parser.add_argument("--num-examples", type=int, default=10, help="분석할 이미지 수 (기본 10장)")
    parser.add_argument("--seed", type=int, default=42, help="이미지 선택 및 W 초기화 시드")
    parser.add_argument("--weight-std", type=float, default=0.01, help="W 초기화 정규분포 표준편차")
    parser.add_argument("--search-size", type=int, default=1000, help="정답/오답 추가 사례 검색 수")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="결과 저장 폴더")
    parser.add_argument("--self-test", action="store_true", help="PPT 수식 검증 (데이터 필요 없음)")
    args = parser.parse_args()

    if args.self_test:
        print(run_formula_self_test())
        return

    # ===== 1. CIFAR-10 test_batch 로딩 =========================================
    # 공식 파일을 그대로 읽는다. 'data_batch_1~5'를 학습에 사용하지 않는다.
    if args.download and not args.data.exists():
        if args.data.suffixes[-2:] != [".tar", ".gz"]:
            parser.error("--download 사용 시 --data에는 .tar.gz 파일 경로를 지정해 주세요.")
        download_cifar10(args.data)
    raw_images, labels = load_cifar10(args.data)
    # raw_images: (10000,3072), labels: (10000,), 픽셀값은 아직 0~255
    print(f"[1] 데이터 로딩 완료: images={raw_images.shape}, labels={labels.shape}")

    # ===== 2. 이미지 10장 선택 =================================================
    # 클래스 0~9에서 각각 1장씩 무작위 선택 (seed 고정) → (10,3072)
    selected_images, selected_labels, selected_indices = select_examples(
        raw_images, labels, args.num_examples, args.seed
    )

    # ===== 3. 픽셀 3072개를 각각 255로 나누어 정규화 ============================
    # X[i,p] = selected_images[i,p] / 255.0 → X: (10,3072), 값의 범위 0~1
    X = preprocess_images(selected_images)
    print(f"[2] 예시 {len(X)}장, 정규화된 입력 X={X.shape}")

    # ===== 4. 클래스별 무작위 W와 b=0 생성 ====================================
    # W[클래스,픽셀]: (10,3072). 클래스마다 다르지만 모든 이미지에 공통 적용.
    W, b = initialize_parameters(args.seed, args.weight_std)

    # ===== 5. 각 이미지에 대해 10개 클래스 점수 계산 ===========================
    # scores[i,j] = Σ_p X[i,p]*W[j,p] + b[j] → scores: (10,10)
    scores = compute_scores(X, W, b)

    # ===== 6. 이미지별 최고 점수의 클래스 번호 선택 ============================
    # predictions[i] = argmax(scores[i,:]) → (10,)
    predictions = predict_classes(scores)
    print(f"[3] 가중치 W={W.shape}, 편향 b={b.shape}, 점수={scores.shape}")

    # ===== 7. SVM Hinge Loss: 정답과 다른 9개 클래스 점수 비교 =================
    # 이미지별 9개 비교 손실을 더한 값: losses_per_image (10,)
    # 그 10개의 손실을 다시 평균: average_loss (스칼라), W는 변경하지 않음.
    losses_per_image, average_loss = compute_svm_loss(scores, selected_labels)
    print(f"[4] Multiclass SVM 평균 Hinge Loss: {average_loss:.6f}")

    # ===== 8. Softmax: SVM Loss가 아니라 원본 scores를 확률로 변환 =============
    # probabilities[i,j] = exp(scores[i,j]) / Σ_k exp(scores[i,k])
    # → probabilities: (10,10), 각 행의 확률 합이 1.
    probabilities = compute_softmax(scores)
    probability_sums, probability_valid = verify_probabilities(probabilities)
    if not probability_valid:
        raise AssertionError("Softmax 확률 합이 1이 아닙니다. 코드를 확인해 주세요.")
    if not np.array_equal(predictions, predict_classes(probabilities)):
        raise AssertionError("Softmax 처리로 예측 클래스가 달라졌습니다.")
    print(f"[5] Softmax 확률 합 검사: {probability_valid}")

    # ===== 9. 별도 탐색 이미지에서 정답/오답 사례를 하나씩 확인 ===================
    # 똑같은 W를 test_batch 앞 1000장에도 적용해 예측과 정답을 비교한다.
    cases = find_correct_incorrect_examples(raw_images, labels, W, b, args.search_size)
    print("[6] 정답 및 오답 사례 탐색 완료")

    # ===== 10. 위에서 계산을 마친 숫자/이미지를 파일로 정리 ====================
    # 아래 함수들은 *계산 결과를 보여주거나 저장*하는 보조 함수들이다.
    # 손실이나 Softmax의 핵심 연산은 이미 5~8단계에서 끝났다.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_score_results(
        selected_indices, selected_labels, predictions, scores, probabilities,
        args.output_dir / "score_probabilities.csv",
    )
    save_loss_results(
        selected_indices, selected_labels, predictions, losses_per_image,
        probability_sums, args.output_dir / "svm_loss_and_probability_sums.csv",
    )
    visualize_predictions(
        selected_images, selected_labels, predictions, selected_indices,
        args.output_dir / "selected_examples.png",
    )
    visualize_correct_incorrect(cases, args.output_dir / "correct_incorrect.png")
    visualize_weights(W, args.output_dir / "random_weights.png")
    visualize_score_softmax_comparison(
        scores[0], probabilities[0], args.output_dir / "score_vs_softmax.png",
    )
    save_explanations(
        args.output_dir / "brief_explanations_ko.md",
        args.seed, args.weight_std, selected_indices, selected_labels, predictions,
        scores, probabilities, losses_per_image, average_loss,
        probability_sums, probability_valid, cases,
    )
    print_detailed_scores(
        selected_indices, selected_labels, predictions, scores, probabilities,
        losses_per_image, probability_sums,
    )
    print("\n" + "=" * 72)
    print(f"평균 SVM Hinge Loss: {average_loss:.6f}")
    print("Softmax 확률 합 모두 1:", probability_valid)
    print(f"[7] 제출용 파일 저장 완료: {args.output_dir.resolve()}")
    print("※ 학습/정확도/정밀도/재현율/F1 계산은 수행하지 않았습니다.")


# -----------------------------------------------------------------------------
# 1. 데이터 불러오기 / 전처리
# -----------------------------------------------------------------------------
def download_cifar10(destination: Path) -> None:
    """공식 CIFAR-10 Python 압축 파일을 다운로드합니다.

    Args:
        destination: 압축 파일을 저장할 경로(예: cifar-10-python.tar.gz).

    Notes:
        --download를 명시적으로 사용한 경우에만 다운로드합니다.
        기존 파일이 있으면 다시 다운로드하지 않습니다.
    """
    if destination.is_file():
        print(f"[데이터] 이미 존재하는 파일 사용: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[데이터] CIFAR-10 다운로드 시작: {CIFAR10_URL}")
    try:
        urllib.request.urlretrieve(CIFAR10_URL, destination)
    except Exception:
        # 중간에 실패한 불완전한 압축 파일을 다음 실행에서 읽지 않도록 삭제합니다.
        destination.unlink(missing_ok=True)
        raise
    print(f"[데이터] 다운로드 완료: {destination}")


def load_cifar10(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """[1단계] test_batch를 읽어 원본 픽셀(아직 0~255)과 정답 번호를 얻는다.

    입력: CIFAR-10 tar.gz 파일 또는 압축 해제된 cifar-10-batches-py 폴더
    출력: images (10000, 3072), labels (10000,)

    핵심: 파일에서 데이터를 읽는 것이지 모델에 데이터를 학습시키는 것이 아니다.
    CIFAR-10 Python 데이터는 pickle이므로 신뢰할 수 있는 공식 배포본만 연다.
    """
    dataset_path = Path(dataset_path)

    # [1-1] 경로가 압축 파일이면 압축을 풀지 않고 그 안의 'test_batch'를 찾는다.
    if dataset_path.is_file():
        with tarfile.open(dataset_path, mode="r:gz") as archive:
            member = archive.extractfile("cifar-10-batches-py/test_batch")
            if member is None:
                raise FileNotFoundError("압축 파일 안에 test_batch가 없습니다.")
            batch = pickle.load(member, encoding="bytes")

    # [1-2] 경로가 폴더이면 해당 폴더 또는 하위 폴더의 'test_batch'를 연다.
    elif dataset_path.is_dir():
        candidates = [
            dataset_path / "test_batch",
            dataset_path / "cifar-10-batches-py" / "test_batch",
        ]
        test_file = next((item for item in candidates if item.is_file()), None)
        if test_file is None:
            raise FileNotFoundError(f"{dataset_path}에서 test_batch를 찾지 못했습니다.")
        with test_file.open("rb") as f:
            batch = pickle.load(f, encoding="bytes")
    else:
        raise FileNotFoundError(f"데이터 파일/폴더가 없습니다: {dataset_path}")

    # [1-3] batch[b'data']: 이미지 10000장의 RGB 픽셀 3072개를 담고 있다.
    #       현재 픽셀값은 0~255 정수이다. 아직 정규화하거나 점수를 구하지 않았다.
    images = np.asarray(batch[b"data"], dtype=np.uint8)

    # [1-4] batch[b'labels']: 각 이미지의 정답 클래스 번호(0~9)이다.
    #       예: 0=airplane, 3=cat, 8=ship (CLASS_NAMES 순서와 일치)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)

    # [1-5] 각 행에는 이미지 1장의 3072개 픽셀이 있어야 한다.
    if images.ndim != 2 or images.shape[1] != INPUT_DIM:
        raise ValueError(f"이미지 모양이 예상과 다릅니다: {images.shape}")
    if labels.shape != (len(images),):
        raise ValueError(f"정답 배열 모양이 예상과 다릅니다: {labels.shape}")

    return images, labels


def select_examples(
    images: np.ndarray,
    labels: np.ndarray,
    num_examples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[2단계] 각 클래스에서 이미지를 뽑는다(기본 10장 = 클래스별 1장).

    단순히 전체 데이터에서 임의의 10장을 뽑는 방식이 아니다.
    기본 설정에서는 airplane~truck의 10개 클래스마다 1장씩 무작위로 선정한다.
    결과의 이미지 행 순서는 클래스 0~9 순서이며, 각 행은 (3072,)이다.

    출력: 선택한 픽셀 (N, 3072), 정답 (N,), 원본 test_batch 인덱스 (N,)
    """
    if not 1 <= num_examples <= len(images):
        raise ValueError("num_examples는 1 이상이며 데이터 수 이하이어야 합니다.")

    # [2-1] 같은 seed를 쓰면 같은 이미지의 인덱스를 선택할 수 있다.
    #       아래 W 초기화 함수 역시 seed=42를 사용하지만 별도의 난수 생성기이다.
    rng = np.random.default_rng(seed)
    chosen_indices = []

    # [2-2] 기본 num_examples=10 → [0,1,2,...,9] 클래스 각각 1개.
    #       예를 들어 num_examples=20 → 각 클래스 2장씩 선택한다.
    wanted_labels = np.arange(num_examples) % NUM_CLASSES
    for class_id in range(NUM_CLASSES):
        count = int(np.sum(wanted_labels == class_id))
        if count == 0:
            continue

        # [2-3] 지금 순회 중인 클래스의 원본 데이터 인덱스를 모은다.
        #       예: class_id=3 → 고양이 이미지만 후보로 추림.
        available_indices = np.flatnonzero(labels == class_id)
        if count > len(available_indices):
            raise ValueError(f"{CLASS_NAMES[class_id]} 클래스의 샘플 수가 부족합니다.")

        # [2-4] 그 클래스의 후보 중 필요한 수만큼 무작위 선택한다.
        #       replace=False: 동일 이미지를 중복 선택하지 않는다.
        sampled_indices = rng.choice(available_indices, size=count, replace=False)
        chosen_indices.extend(sampled_indices.tolist())

    # [2-5] 발표와 출력 표를 읽기 쉽도록 정답 클래스 번호 순으로 정렬한다.
    chosen_indices = np.asarray(chosen_indices, dtype=np.int64)
    chosen_indices = chosen_indices[np.argsort(labels[chosen_indices], kind="stable")]

    # 이미지의 픽셀과 정답, 원본 test_batch 인덱스를 나란히 반환한다.
    return images[chosen_indices], labels[chosen_indices], chosen_indices


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """[3단계] N장의 3072개 픽셀값을 0~255에서 0~1 사이로 변경한다.

    계산: x_normalized = x_original / 255.0
    예: 픽셀 0→0.0, 128→약 0.502, 255→1.0
    출력: (N,3072) 모양의 실수형 X (입력의 이미지 순서 그대로)

    참고: CIFAR-10 Python 파일은 이미 (N,3072)로 저장되므로 여기서는
    32×32×3을 다시 펼치는 연산이 필요 없다. 계산하는 동안 RGB 순서는 유지된다.
    """
    if images.ndim != 2 or images.shape[1] != INPUT_DIM:
        raise ValueError(f"예상 입력 크기: (N, {INPUT_DIM}), 실제: {images.shape}")

    # [3-1] 소수점 계산을 위해 정수(uint8) 픽셀을 실수(float64)로 변환한다.
    images_as_float = images.astype(np.float64)

    # [3-2] 픽셀값 하나하나를 255로 나눠 선형 분류기에 입력할 X를 만든다.
    #       N=10이라면 X.shape=(10,3072). 이미지당 3072개의 특징은 그대로다.
    X = images_as_float / 255.0
    return X


def flat_to_rgb_image(flat_image: np.ndarray) -> np.ndarray:
    """그림을 표시할 때만 3072차원 1개 벡터를 32×32×3 RGB로 복원한다.

    CIFAR-10 파일은 R 채널 1024개 + G 채널 1024개 + B 채널 1024개 순서이다.
    따라서 그냥 reshape(32,32,3)하면 RGB 위치가 뒤섞인다.
    분류 계산용 X나 W 값은 변경하지 않고 '보여주는 모양'만 변환한다.
    """
    # 3072 → (3,32,32): 첫 1024개를 R, 다음 1024개를 G, 마지막을 B로 놓는다.
    channel_first = flat_image.reshape(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)

    # (3,32,32) → (32,32,3): matplotlib이 기대하는 (높이,너비,RGB) 순서.
    rgb_image = channel_first.transpose(1, 2, 0)
    return rgb_image


# -----------------------------------------------------------------------------
# 2. 모델 파라미터 초기화 / 클래스 점수 계산
# -----------------------------------------------------------------------------
def initialize_parameters(
    seed: int = 42,
    weight_std: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """[4단계] 학습되지 않은 가중치 W(10,3072)와 편향 b(10,)를 만든다.

    - W의 행 0: airplane 점수용 가중치 3072개
    - W의 행 3: cat 점수용 가중치 3072개
    - W의 행 9: truck 점수용 가중치 3072개

    각 클래스는 서로 다른 무작위 가중치를 갖는다. 단, 이미지마다 새로운 W를
    생성하는 것이 아니라, 여기서 한 번 만든 '동일한 W'를 모든 이미지에 쓴다.
    """
    if weight_std <= 0:
        raise ValueError("weight_std는 0보다 커야 합니다.")

    # [4-1] seed=42 고정 → 다시 실행해도 똑같은 무작위 수를 생성한다.
    rng = np.random.default_rng(seed)

    # [4-2] 평균 0, 표준편차 0.01인 정규분포에서 10×3072개 숫자를 생성한다.
    #       W[class_id, pixel_id]가 '특정 클래스'와 '특정 픽셀'의 가중치이다.
    #       여기에는 정답/이미지 정보가 쓰이지 않는다 → 무작위 초기화만 수행.
    W = rng.normal(loc=0.0, scale=weight_std, size=(NUM_CLASSES, INPUT_DIM))

    # [4-3] b[class_id]는 클래스별 상수 항이다. 과제 조건에 따라 전부 0으로 설정.
    b = np.zeros(NUM_CLASSES, dtype=np.float64)
    return W, b


def compute_scores(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[5단계] 각 이미지 x에 같은 W, b를 적용해 클래스별 점수 10개를 계산한다.

    이미지 1장: x(3072,) → W(10,3072) @ x(3072,) + b(10,) → 점수(10,)
    이미지 N장: X(N,3072) @ W.T(3072,10) + b(10,) → scores(N,10)

    예: scores[0,5]는 '첫 번째 이미지에 대한 dog 클래스 점수'이다.
    이 점수는 아직 확률이 아니고 음수가 나올 수도 있다.
    """
    if X.ndim != 2 or X.shape[1] != INPUT_DIM:
        raise ValueError("X는 (N, 3072) 형태여야 합니다.")
    if W.shape != (NUM_CLASSES, INPUT_DIM) or b.shape != (NUM_CLASSES,):
        raise ValueError("W는 (10, 3072), b는 (10,) 형태여야 합니다.")

    # [5-1] X의 행 i = i번째 이미지 3072개 픽셀.
    #       W의 행 j = j번째 클래스가 픽셀 3072개에 부여하는 가중치.
    #       scores[i,j] = Σ_(pixel=0..3071) X[i,pixel]*W[j,pixel] + b[j]
    #
    #       단일 이미지의 수식은 f(x)=Wx+b이지만, 여러 이미지를 행으로 쌓은
    #       X의 차원은 (N,3072)이므로 파이썬에서는 W를 전치한 W.T를 곱한다.
    #       이 한 줄이 모든 이미지×모든 클래스의 '내적'을 동시에 수행한다.
    scores = X @ W.T + b

    # [5-2] 이미지 10장 → 10×10 점수 행렬. W는 변경되지 않는다.
    return scores


def predict_classes(scores: np.ndarray) -> np.ndarray:
    """[6단계] 이미지마다 최고 점수를 받은 클래스의 번호(0~9)를 선택한다.

    scores의 한 행: [airplane점수, automobile점수, ..., truck점수]
    argmax(axis=1): 각 '이미지 행'에서 가장 큰 점수가 있는 '클래스 열'의 번호
    숫자 예측값은 CLASS_NAMES[번호]로 사람이 읽는 클래스 이름으로 바꿀 수 있다.
    """
    # 예: 첫 이미지의 최대 점수가 scores[0,5]면 predictions[0]=5 → dog.
    predictions = np.argmax(scores, axis=1)
    return predictions


# -----------------------------------------------------------------------------
# 3. Multiclass SVM Hinge Loss / Softmax
# -----------------------------------------------------------------------------
def compute_svm_loss(
    scores: np.ndarray,
    true_labels: np.ndarray,
    delta: float = 1.0,
) -> tuple[np.ndarray, float]:
    """[7단계] 정답 점수를 다른 9개 클래스 점수와 비교해 SVM 손실을 계산한다.

    수식 (강의 Week 9-2): L_i = Σ_(j≠y_i) max(0, s_ij - s_i,y_i + 1)
       s_i,y_i = 이미지 i의 '정답 클래스 점수',  s_ij = 이미지 i의 '비교 클래스 점수'

    10개 클래스 중 정답 1개는 합산 대상에서 제외 → 이미지당 9개 비교.
    이미지 1장 손실 = 9개 비교 손실의 합 (9로 평균 내지 않음).
    전체 평균 손실 = N장 각각의 손실값을 더한 다음 N으로 나눈 값.

    실제 첫 번째 예시: airplane 점수 -0.353013, dog 점수 0.867669
      dog와 비교한 손실 = max(0, 0.867669 - (-0.353013) + 1) = 2.220682
    모든 비교 항목을 합하면 약 14.642571. 정답이어도 margin 1 미만이면 Loss>0.

    이 함수는 '얼마나 잘못 점수를 매겼는가'만 측정하며, W는 수정하지 않는다.
    """
    n = scores.shape[0]
    if n == 0 or scores.shape != (n, NUM_CLASSES):
        raise ValueError("scores는 (N, 10) 형태이며 N은 1 이상이어야 합니다.")
    if true_labels.shape != (n,):
        raise ValueError("true_labels는 (N,) 형태여야 합니다.")
    if delta <= 0:
        raise ValueError("delta는 0보다 커야 합니다.")

    # [7-1] 이미지 i마다 정답 클래스 번호 y_i를 이용해 '정답의 점수'를 뽑는다.
    #       예: 첫 이미지의 정답=airplane(번호 0) → scores[0,0]=-0.353013.
    #       결과 모양 (N,)을 (N,1)로 바꾸면 각 이미지의 10개 점수에서
    #       해당 이미지의 정답 점수를 한꺼번에 뺄 수 있다(브로드캐스팅).
    correct_class_scores = scores[np.arange(n), true_labels][:, np.newaxis]

    # [7-2] 각 이미지의 10개 후보 점수에 대해 s_j - s_y + margin(=1) 계산.
    #       max(0,...)으로 음수는 0 처리, 양수만 손실로 남긴다.
    #       예: dog와의 비교 = max(0,0.867669 - (-0.353013) + 1).
    margins = np.maximum(0.0, scores - correct_class_scores + delta)

    # [7-3] 정답 클래스 자기 자신과 비교한 값은 제외!
    #       제외하지 않으면 s_y-s_y+1 = 1이 불필요하게 더해진다.
    margins[np.arange(n), true_labels] = 0.0

    # [7-4] 이미지별로 '정답 이외의 9개 클래스 손실'을 모두 더한다.
    #       axis=1: 한 이미지(한 행)의 10개 열을 합산 → (N,) 벡터.
    losses_per_image = margins.sum(axis=1)

    # [7-5] 위에서 구한 이미지 N장의 손실을 평균 낸다.
    #       선택된 10장에서 관찰된 값: 평균 8.909212.
    average_loss = float(np.mean(losses_per_image))
    return losses_per_image, average_loss


def compute_softmax(scores: np.ndarray) -> np.ndarray:
    """[8단계] SVM Loss가 아니라 '원본 10개 점수'를 Softmax에 넣는다.

    수식: P_ij = exp(s_ij) / Σ_k exp(s_ik)
    입력 scores(N,10) → 출력 probabilities(N,10)
    이미지마다 10개 값이 모두 0~1이고 그 합은 1이 된다.

    중요한 점: SVM Loss와 Softmax는 각각 '같은 원본 scores'에서 갈라진다.
    Softmax는 W를 학습하지 않으며, 점수의 순위(argmax)를 바꾸지 않는다.
    """
    if scores.ndim != 2 or scores.shape[1] != NUM_CLASSES:
        raise ValueError("scores는 (N, 10) 형태여야 합니다.")

    # [8-1] 이미지(행)마다 10개 점수 중 가장 큰 점수를 찾은 뒤 전부에서 뺀다.
    #       예: 최대 0.867669이면 모든 점수에서 0.867669를 뺀다.
    #       exp(큰 양수)의 overflow 방지용으로, 최종 확률은 달라지지 않는다.
    #       keepdims=True → 최대 점수 결과 (N,1), 이미지별로 뺄 수 있음.
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)

    # [8-2] 각 클래스 점수에 e^점수를 적용한다 → 10개 값 모두 양수.
    exp_scores = np.exp(shifted_scores)

    # [8-3] 각 이미지의 10개 e^점수를 합하고, 개별 값을 이 합으로 나눈다.
    #       예: probabilities[0,5]는 첫 번째 이미지의 dog Softmax 값.
    #       각 이미지의 10개 값을 합하면 1이 된다.
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probabilities


def verify_probabilities(probabilities: np.ndarray) -> tuple[np.ndarray, bool]:
    """[9단계] N장 각각의 클래스 확률 10개를 합하여 1인지 확인한다."""
    # axis=1: 이미지별로 클래스 10개를 합산 → N개 합계가 나온다.
    sums = probabilities.sum(axis=1)

    # 컴퓨터의 실수 계산에는 작은 오차가 있으므로 == 1이 아닌 allclose 사용.
    all_valid = bool(np.allclose(sums, 1.0, rtol=1e-10, atol=1e-10))
    return sums, all_valid


# -----------------------------------------------------------------------------
# 4. 올바른 예측 / 잘못된 예측 사례 찾기
# -----------------------------------------------------------------------------
def find_correct_incorrect_examples(
    all_images: np.ndarray,
    all_labels: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    search_size: int = 1000,
) -> dict[str, dict | None]:
    """[10단계] 같은 W를 적용해 정답 1장과 오답 1장을 찾아 비교한다.

    앞서 선택한 10장에 정답이 반드시 있을지는 알 수 없다. 그래서 별도로
    test_batch의 '앞에서부터 최대 1000장'을 살펴본다(랜덤 재선택 아님).
    이 과정은 비교를 위한 사례 탐색이며 모델 학습은 아니다.
    """
    count = min(max(1, search_size), len(all_images))

    # [10-1] 별도의 1000장을 255로 나눠 X_search(1000,3072)를 만든다.
    X_search = preprocess_images(all_images[:count])

    # [10-2] 앞에서 초기화한 '동일한 W,b'로 1000장의 점수와 예측값을 계산한다.
    #        정답 예시를 고른다고 W를 학습하거나 변경하는 것은 아니다.
    search_scores = compute_scores(X_search, W, b)
    search_predictions = predict_classes(search_scores)

    # [10-3] 각 이미지에서 예측 번호와 실제 라벨 번호가 같은지 확인한다.
    matches = search_predictions == all_labels[:count]

    # [10-4] 첫 번째 정답(True)과 첫 번째 오답(False)의 원본 인덱스 기록.
    cases: dict[str, dict | None] = {"correct": None, "incorrect": None}
    for case_name, match_value in (("correct", True), ("incorrect", False)):
        positions = np.flatnonzero(matches == match_value)
        if len(positions) > 0:
            idx = int(positions[0])
            cases[case_name] = {
                "index": idx,
                "image": all_images[idx],
                "true_label": int(all_labels[idx]),
                "predicted_label": int(search_predictions[idx]),
            }
    return cases


# -----------------------------------------------------------------------------
# 5. 시각화 (그림은 전부 PNG 파일로 저장)
# -----------------------------------------------------------------------------
def visualize_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    image_indices: np.ndarray,
    output_path: Path,
) -> None:
    """선택한 실제 이미지를 보여주고 정답(True)과 예측(Pred)을 표시합니다."""
    count = len(images)
    cols = min(5, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.1), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")
    for i, ax in enumerate(axes.flat[:count]):
        ax.imshow(flat_to_rgb_image(images[i]))
        status = "CORRECT" if true_labels[i] == predictions[i] else "WRONG"
        ax.set_title(
            f"Test #{int(image_indices[i])} / {status}\n"
            f"True: {CLASS_NAMES[true_labels[i]]}\n"
            f"Pred: {CLASS_NAMES[predictions[i]]}",
            fontsize=10,
        )
    fig.suptitle("CIFAR-10: predictions with randomly initialized weights", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_correct_incorrect(cases: dict[str, dict | None], output_path: Path) -> None:
    """맞춘 예시 1장과 틀린 예시 1장을 비교해 보여줍니다."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (key, title) in zip(axes, (("correct", "Correct example"), ("incorrect", "Incorrect example"))):
        item = cases[key]
        ax.axis("off")
        if item is None:
            ax.text(0.5, 0.5, "Not found in search pool", ha="center", va="center")
        else:
            ax.imshow(flat_to_rgb_image(item["image"]))
            ax.set_title(
                f"{title} (Test #{item['index']})\n"
                f"True: {CLASS_NAMES[item['true_label']]}\n"
                f"Pred: {CLASS_NAMES[item['predicted_label']]}"
            )
    fig.suptitle("Correct vs. incorrect (no training)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_weights(W: np.ndarray, output_path: Path) -> None:
    """[11단계] '클래스별 가중치 3072개'를 RGB 이미지처럼 배열하여 관찰한다.

    여기서 시각화하는 것은 입력 이미지 x가 아니라 W의 각 행이다.
      W[0] (3072,) → airplane에 대응하는 가중치의 RGB 패턴
      W[3] (3072,) → cat에 대응하는 가중치의 RGB 패턴
    학습하지 않았으므로 실제 비행기/고양이 형상이 아닌 잡음처럼 보인다.
    """
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))

    # [11-1] W에는 음수도 있으므로 이미지로 표시할 때만 0~1로 변환할 기준을 구한다.
    #        W 원본의 값 자체는 전혀 수정하지 않는다.
    visual_scale = 3.0 * float(np.std(W))

    for class_id, ax in enumerate(axes.flat):
        # [11-2] W[class_id] 3072개를 (32,32,3) RGB 순서로 재배치한다.
        weight_image = flat_to_rgb_image(W[class_id])

        # [11-3] 화면 표시 전용: 0에 가까운 가중치→회색(0.5),
        #        양수→밝게, 음수→어둡게 나타나도록 정규화/클리핑한다.
        display_image = np.clip(weight_image / (2 * visual_scale) + 0.5, 0.0, 1.0)

        # [11-4] class_id에 해당하는 이름과 RGB 가중치 패턴을 그린다.
        ax.imshow(display_image)
        ax.set_title(CLASS_NAMES[class_id])
        ax.axis("off")

    fig.suptitle("Random weight templates: noise, not learned class features")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_score_softmax_comparison(
    scores_for_one_image: np.ndarray,
    probabilities_for_one_image: np.ndarray,
    output_path: Path,
) -> None:
    """첫 번째 예시의 원본 점수와 Softmax 확률을 나란히 비교합니다."""
    x_positions = np.arange(NUM_CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(x_positions, scores_for_one_image)
    axes[0].set_title("Raw class scores (can be negative)")
    axes[0].set_ylabel("Linear score")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[1].bar(x_positions, probabilities_for_one_image)
    axes[1].set_title("After Softmax (sum = 1)")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0.0, max(0.2, float(np.max(probabilities_for_one_image)) * 1.2))
    for ax in axes:
        ax.set_xticks(x_positions, CLASS_NAMES, rotation=45, ha="right")
        ax.set_xlabel("Class")
    fig.suptitle("Same prediction, different score representations")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 6. 실행 결과를 파일(CSV, Markdown)로 저장
# -----------------------------------------------------------------------------
def save_score_results(
    image_indices: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """이미지마다 10개 클래스 전체의 원본 점수/Softmax 확률을 CSV로 저장합니다.

    예: 10장 * 10클래스 = 총 100행의 결과가 저장됩니다.
    정답과 예측이 맞았는지도 함께 기록합니다.
    """
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "example_no", "test_index", "true_label", "predicted_label",
            "correct", "class_name", "raw_score", "softmax_probability",
        ])
        for i in range(len(image_indices)):
            for class_id in range(NUM_CLASSES):
                writer.writerow([
                    i + 1,
                    int(image_indices[i]),
                    CLASS_NAMES[true_labels[i]],
                    CLASS_NAMES[predictions[i]],
                    bool(true_labels[i] == predictions[i]),
                    CLASS_NAMES[class_id],
                    f"{scores[i, class_id]:.8f}",
                    f"{probabilities[i, class_id]:.8f}",
                ])


def save_loss_results(
    image_indices: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    losses_per_image: np.ndarray,
    probability_sums: np.ndarray,
    output_path: Path,
) -> None:
    """이미지별 SVM Loss 및 확률의 합을 CSV로 저장합니다."""
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "example_no", "test_index", "true_label", "predicted_label",
            "correct", "svm_hinge_loss", "softmax_probability_sum",
        ])
        for i in range(len(image_indices)):
            writer.writerow([
                i + 1,
                int(image_indices[i]),
                CLASS_NAMES[true_labels[i]],
                CLASS_NAMES[predictions[i]],
                bool(true_labels[i] == predictions[i]),
                f"{losses_per_image[i]:.8f}",
                f"{probability_sums[i]:.12f}",
            ])


def save_explanations(
    output_path: Path,
    seed: int,
    weight_std: float,
    image_indices: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    probabilities: np.ndarray,
    losses_per_image: np.ndarray,
    average_loss: float,
    probability_sums: np.ndarray,
    probability_valid: bool,
    cases: dict[str, dict | None],
) -> None:
    """과제에 필요한 계산 결과와 짧은 한국어 설명을 Markdown 파일로 저장합니다."""
    lines = [
        "# CIFAR-10 Linear Classification 실험 결과",
        "",
        "## 1. 실험 설정",
        "",
        f"- 예시 이미지 수: {len(image_indices)}장 (CIFAR-10 test_batch)",
        f"- 난수 시드: `{seed}`",
        f"- 가중치: `W.shape = (10, 3072)`, 정규분포 N(0, {weight_std}²)",
        "- 편향: `b.shape = (10,)`, 모든 원소 0",
        "- 픽셀: 0~255에서 0~1로 정규화",
        "- **학습, 역전파, 가중치 업데이트를 수행하지 않음**",
        "",
        "## 2. 점수 및 예측",
        "",
        "각 이미지를 3072차원 벡터로 만들고 `f(x) = Wx + b`로 10개 클래스 점수를 계산했습니다.",
        "가장 높은 점수를 갖는 클래스(`argmax`)를 예측 결과로 사용했습니다.",
        "점수는 음수가 될 수 있고, 확률이 아닙니다.",
        "",
        "| 예시 | 원본 Test 인덱스 | 정답 | 예측 | 일치 여부 | SVM Loss | 확률 합 |",
        "|---:|---:|---|---|---|---:|---:|",
    ]
    for i in range(len(image_indices)):
        correct_text = "정답" if true_labels[i] == predictions[i] else "오답"
        lines.append(
            f"| {i+1} | {int(image_indices[i])} | {CLASS_NAMES[true_labels[i]]} | "
            f"{CLASS_NAMES[predictions[i]]} | {correct_text} | "
            f"{losses_per_image[i]:.6f} | {probability_sums[i]:.12f} |"
        )

    lines.extend([
        "",
        "## 3. Multiclass SVM Hinge Loss",
        "",
        "수식: `L_i = Σ_{j≠y_i} max(0, s_j - s_y + 1)`",
        "",
        f"**선택한 {len(image_indices)}장에 대한 평균 손실: {average_loss:.6f}**",
        "",
        "점수가 정답 클래스에서 가장 높더라도, 다른 클래스보다 1 이상 높지 않으면 ",
        "SVM 손실이 0보다 클 수 있습니다. 가중치가 무작위이므로 큰 손실도 자연스럽습니다.",
        "",
        "## 4. Softmax 계산",
        "",
        "수식: `P_j = exp(s_j - max(s)) / Σ_k exp(s_k - max(s))`",
        "",
        f"- 선택한 모든 이미지의 확률 합이 1인가? **{probability_valid}**",
        "- 각 이미지의 10개 확률 합은 위 표에서 확인할 수 있습니다.",
        "- Softmax는 점수를 양수 확률 형태로 바꾸지만, 클래스 순위(argmax)는 바꾸지 않습니다.",
        "- 학습 전 결과이므로 이 값이 의미 있는 모델의 신뢰도를 뜻하지는 않습니다.",
        "",
        "### 예시 이미지 1의 모든 클래스 점수와 Softmax 확률",
        "",
        "| 클래스 | 원본 점수 | Softmax 확률 |",
        "|---|---:|---:|",
    ])
    for class_id, name in enumerate(CLASS_NAMES):
        lines.append(f"| {name} | {scores[0, class_id]:.6f} | {probabilities[0, class_id]:.6f} |")

    lines.extend(["", "## 5. 올바른 분류 / 잘못된 분류 사례", ""])
    for case_name, korean_name in (("correct", "올바른 분류"), ("incorrect", "잘못된 분류")):
        item = cases[case_name]
        if item is None:
            lines.append(f"- {korean_name}: 설정한 탐색 범위 안에서 발견되지 않았습니다.")
        else:
            lines.append(
                f"- {korean_name}: Test #{item['index']}, "
                f"정답 `{CLASS_NAMES[item['true_label']]}`, "
                f"예측 `{CLASS_NAMES[item['predicted_label']]}`."
            )
    lines.extend([
        "",
        "가중치를 학습하지 않았으므로 맞는 예측도 우연일 가능성이 높고, "
        "틀린 예측은 이미지별 특징을 아직 학습하지 못한 결과입니다.",
        "",
        "## 6. 가중치 시각화 설명",
        "",
        "`random_weights.png`는 각 클래스의 가중치 3072개를 `32 × 32 × 3`으로 복원한 그림입니다.",
        "클래스마다 서로 다른 무작위 수를 가진 것이므로 뚜렷한 비행기, 고양이, "
        "자동차 등의 모양이 아니라 잡음과 비슷한 패턴이 나타납니다.",
        "학습했다면 데이터와 정답을 이용해 W가 조정될 수 있지만, 이 과제에서는 그런 과정이 없습니다.",
        "",
        "## 7. 제출 파일",
        "",
        "- `score_probabilities.csv`: 모든 예시의 클래스 10개 점수와 확률",
        "- `svm_loss_and_probability_sums.csv`: 개별 SVM Loss 및 확률 합",
        "- `selected_examples.png`: 정답과 예측을 표시한 원본 이미지",
        "- `correct_incorrect.png`: 맞은 사례와 틀린 사례 비교",
        "- `score_vs_softmax.png`: 동일한 이미지의 점수와 Softmax 확률 비교",
        "- `random_weights.png`: 학습 전 10개 클래스의 무작위 가중치 이미지",
        "",
        "※ Accuracy/Precision/Recall/F1 등의 평가지표는 계산하지 않았습니다.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_detailed_scores(
    image_indices: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    probabilities: np.ndarray,
    losses: np.ndarray,
    sums: np.ndarray,
) -> None:
    """터미널에서도 각 예시의 10개 점수와 확률을 전부 확인할 수 있게 출력합니다."""
    for i in range(len(image_indices)):
        print("\n" + "=" * 72)
        print(f"예시 {i+1} / CIFAR Test #{int(image_indices[i])}")
        print(f"정답: {CLASS_NAMES[true_labels[i]]} | 예측: {CLASS_NAMES[predictions[i]]}")
        print(f"결과: {'정답' if true_labels[i] == predictions[i] else '오답'}")
        print(f"SVM Hinge Loss: {losses[i]:.6f}")
        print(f"{'Class':<13}{'Raw score':>14}{'Softmax':>15}")
        for class_id, name in enumerate(CLASS_NAMES):
            print(f"{name:<13}{scores[i, class_id]:>14.6f}{probabilities[i, class_id]:>15.6f}")
        print(f"10개 Softmax 확률의 합: {sums[i]:.12f}")


# -----------------------------------------------------------------------------
# 7. 계산 검증: 강의 PPT의 예제를 이용한 자체 테스트
# -----------------------------------------------------------------------------
def run_formula_self_test() -> str:
    """PPT 23~25페이지의 예시를 사용해 SVM Loss 구현을 검증합니다.

    이미지 1: 정답 cat, 점수 [cat=3.2, car=5.1, frog=-1.7] -> 2.9
    이미지 2: 정답 car, 점수 [cat=1.3, car=4.9, frog=2.0] -> 0.0
    이미지 3: 정답 frog, 점수 [cat=2.2, car=2.5, frog=-3.1] -> 12.9

    함수가 10개 클래스를 전제로 하므로, 이 자체 테스트에서는 별도의
    3클래스용 참조 구현을 작성하여 같은 수식을 직접 확인합니다.
    """
    # 실제 강의 예시를 사용합니다. 이 값은 CIFAR-10 예측 결과가 아닙니다.
    scores = np.array([
        [3.2, 5.1, -1.7],
        [1.3, 4.9, 2.0],
        [2.2, 2.5, -3.1],
    ], dtype=np.float64)
    labels = np.array([0, 1, 2])

    def reference_svm_loss(values: np.ndarray, label: int) -> float:
        """3클래스에서도 동작하는, 수식을 그대로 옮긴 작은 검사 함수."""
        correct_score = values[label]
        return float(sum(
            max(0.0, float(score - correct_score + 1.0))
            for class_id, score in enumerate(values) if class_id != label
        ))

    observed = np.array([reference_svm_loss(row, label) for row, label in zip(scores, labels)])
    expected = np.array([2.9, 0.0, 12.9])
    assert np.allclose(observed, expected), f"SVM 테스트 실패: {observed}"

    # 본 과제의 10클래스용 손실 함수도 별도의 독립된 점수로 검증합니다.
    padded_scores = np.full((3, 10), -100.0)
    padded_scores[:, :3] = scores
    actual, mean_loss = compute_svm_loss(padded_scores, labels)
    assert np.allclose(actual, expected), f"10클래스 SVM 테스트 실패: {actual}"
    assert np.isclose(mean_loss, expected.mean())

    # 오버플로우가 일어날 수 있는 큰 점수에 대해서도 합이 1인지 확인합니다.
    big_scores = np.array([[1000.0] + [999.0] * 9])
    probability_sums, valid = verify_probabilities(compute_softmax(big_scores))
    assert valid and np.isclose(probability_sums[0], 1.0)

    return (
        "PPT 수식 검증 통과 (실제 CIFAR-10 예측 결과가 아님)\n"
        f"이미지 1 SVM Loss = {actual[0]:.6f} (기대값 2.9)\n"
        f"이미지 2 SVM Loss = {actual[1]:.6f} (기대값 0.0)\n"
        f"이미지 3 SVM Loss = {actual[2]:.6f} (기대값 12.9)\n"
        f"평균 Loss = {mean_loss:.6f}\n"
        f"큰 점수에 대한 Softmax 합 = {probability_sums[0]:.12f}\n"
    )


# -----------------------------------------------------------------------------
# 8. 전체 실행을 관리하는 main 함수
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
