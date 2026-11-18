"""CIFAR-10 선형 분류기 과제 (학습 없이 무작위 가중치 사용).

참고: Week 9-2 Linear Classification 강의자료
실행 예시:
    pip install -r requirements.txt
    python linear_classification.py --data cifar-10-python.tar.gz
    python linear_classification.py --download
    python linear_classification.py --self-test

이 프로그램은 가중치의 업데이트, 최적화, 모델 학습을 수행하지 않습니다.
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
    """CIFAR-10의 test_batch만 불러옵니다. **학습 데이터는 사용하지 않습니다.**

    Args:
        dataset_path:
            1) 공식 파일 cifar-10-python.tar.gz 또는
            2) 압축을 푼 cifar-10-batches-py 폴더 또는 그 상위 폴더.

    Returns:
        images: (10000, 3072) 형태의 uint8 배열. RGB 채널별로 펼쳐진 픽셀 값.
        labels: (10000,) 형태의 정답 번호(0~9).

    주의:
        공식 CIFAR-10 Python 데이터는 pickle 파일입니다. 신뢰할 수 있는
        공식 배포본만 불러오세요. 출처를 알 수 없는 pickle은 사용하지 마세요.
    """
    dataset_path = Path(dataset_path)

    if dataset_path.is_file():
        # 압축을 풀지 않아도 tar.gz 내부의 test_batch 파일을 직접 읽습니다.
        with tarfile.open(dataset_path, mode="r:gz") as archive:
            target = "cifar-10-batches-py/test_batch"
            member = archive.extractfile(target)
            if member is None:
                raise FileNotFoundError(f"압축 파일에서 '{target}'를 찾을 수 없습니다.")
            batch = pickle.load(member, encoding="bytes")

    elif dataset_path.is_dir():
        # 압축을 미리 풀어둔 경우에는 해당 폴더에서도 읽을 수 있습니다.
        candidates = [
            dataset_path / "test_batch",
            dataset_path / "cifar-10-batches-py" / "test_batch",
        ]
        test_file = next((path for path in candidates if path.is_file()), None)
        if test_file is None:
            raise FileNotFoundError(
                f"{dataset_path} 안에 test_batch가 없습니다. "
                "cifar-10-batches-py 폴더를 지정해 주세요."
            )
        with test_file.open("rb") as f:
            batch = pickle.load(f, encoding="bytes")
    else:
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {dataset_path}\n"
            "공식 압축 파일을 코드와 같은 폴더에 두거나 --download 옵션을 사용하세요."
        )

    # CIFAR-10 Python 배포본의 key는 bytes 타입일 수 있습니다.
    images = np.asarray(batch[b"data"], dtype=np.uint8)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)

    # 모양을 확인하면 파일을 잘못 불러오는 실수를 빠르게 찾을 수 있습니다.
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
    """10개 클래스가 고르게 포함되도록 예시 이미지를 고정 시드로 뽑습니다.

    예: 10장 선택 -> 클래스 0~9에서 각각 한 장씩 무작위 선택.
    예: 20장 선택 -> 각 클래스에서 두 장씩 선택.

    클래스별 샘플을 고르게 선정하는 것은 '시각화용 예시 선택'이며,
    가중치 학습이나 좋은 예측을 유도하는 절차가 아닙니다.

    Returns:
        selected_images: (N, 3072) 원본 픽셀
        selected_labels: (N,) 실제 정답
        selected_indices: (N,) 원본 test_batch에서의 이미지 인덱스
    """
    if not 1 <= num_examples <= len(images):
        raise ValueError("num_examples는 1 이상이며 데이터 수 이하이어야 합니다.")

    rng = np.random.default_rng(seed)
    chosen_indices = []

    # 10장 미만이면 앞의 일부 클래스만 포함되는 방식입니다.
    # 10장 이상이면 정확히 10개 클래스가 한 번씩 포함됩니다.
    wanted_labels = np.arange(num_examples) % NUM_CLASSES
    for class_id in range(NUM_CLASSES):
        count = int(np.sum(wanted_labels == class_id))
        if count == 0:
            continue
        available_indices = np.flatnonzero(labels == class_id)
        if count > len(available_indices):
            raise ValueError(f"{CLASS_NAMES[class_id]} 클래스의 샘플 수가 부족합니다.")
        sampled_indices = rng.choice(available_indices, size=count, replace=False)
        chosen_indices.extend(sampled_indices.tolist())

    # 클래스 순서대로 결과를 볼 수 있도록 정답 번호로 정렬합니다.
    chosen_indices = np.asarray(chosen_indices, dtype=np.int64)
    chosen_indices = chosen_indices[np.argsort(labels[chosen_indices], kind="stable")]
    return images[chosen_indices], labels[chosen_indices], chosen_indices


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """픽셀을 0~1 범위로 정규화하여 (N, 3072) 실수 배열로 바꿉니다.

    원본 픽셀 값: 0~255 -> 정규화한 값: 0.0~1.0
    CIFAR-10의 저장 형태는 이미 (N, 3072)로 펼쳐져 있습니다.
    원래 이미지가 (N, 32, 32, 3)이라면 채널별 순서에 맞게 펼쳐야 합니다.

    주의: 여기서는 간단한 입력값 스케일 조정만 수행합니다.
    데이터 통계로 모델을 학습하거나 가중치를 변경하지 않습니다.
    """
    if images.ndim != 2 or images.shape[1] != INPUT_DIM:
        raise ValueError(f"예상 입력 크기: (N, {INPUT_DIM}), 실제: {images.shape}")
    return images.astype(np.float64) / 255.0


def flat_to_rgb_image(flat_image: np.ndarray) -> np.ndarray:
    """CIFAR-10의 3072개 픽셀을 사람이 보는 RGB 이미지로 복원합니다.

    공식 CIFAR-10 파일은 R 1024개, G 1024개, B 1024개 순서로 저장합니다.
    따라서 (3, 32, 32)로 만들고 채널을 마지막 축으로 옮겨야 합니다.
    일반적인 reshape(32, 32, 3)만 하면 채널 순서가 뒤섞입니다.
    """
    return flat_image.reshape(3, IMAGE_HEIGHT, IMAGE_WIDTH).transpose(1, 2, 0)


# -----------------------------------------------------------------------------
# 2. 모델 파라미터 초기화 / 클래스 점수 계산
# -----------------------------------------------------------------------------
def initialize_parameters(
    seed: int = 42,
    weight_std: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """학습 전 상태를 만들기 위해 작은 무작위 가중치와 0 편향을 생성합니다.

    W: (10, 3072) -> 각 행은 해당 클래스의 3072개 픽셀 가중치
    b: (10,)      -> 각 클래스별 편향 하나

    같은 seed를 사용하면 같은 W와 b를 얻으므로 결과를 재현할 수 있습니다.
    """
    if weight_std <= 0:
        raise ValueError("weight_std는 0보다 커야 합니다.")
    rng = np.random.default_rng(seed)
    W = rng.normal(loc=0.0, scale=weight_std, size=(NUM_CLASSES, INPUT_DIM))
    b = np.zeros(NUM_CLASSES, dtype=np.float64)
    return W, b


def compute_scores(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """선형 분류기의 핵심 함수 f(x) = Wx + b를 계산합니다.

    단일 이미지의 수식:
        W(10, 3072) @ x(3072,) + b(10,) -> score(10,)

    여러 이미지의 NumPy 벡터화 계산:
        X(N, 3072) @ W.T(3072, 10) + b(10,) -> scores(N, 10)

    점수(score)는 확률이 아닙니다. 음수일 수도 있고 합이 1일 필요도 없습니다.
    """
    if X.ndim != 2 or X.shape[1] != INPUT_DIM:
        raise ValueError("X는 (N, 3072) 형태여야 합니다.")
    if W.shape != (NUM_CLASSES, INPUT_DIM) or b.shape != (NUM_CLASSES,):
        raise ValueError("W는 (10, 3072), b는 (10,) 형태여야 합니다.")
    return X @ W.T + b


def predict_classes(scores: np.ndarray) -> np.ndarray:
    """각 이미지에서 가장 점수가 높은 클래스의 인덱스를 반환합니다.

    예: scores = [0.2, -0.4, 0.8, ...] -> 가장 큰 0.8의 위치를 예측
    argmax(axis=1)는 이미지별(행별) 최고 점수의 열 번호를 반환합니다.
    """
    return np.argmax(scores, axis=1)


# -----------------------------------------------------------------------------
# 3. Multiclass SVM Hinge Loss / Softmax
# -----------------------------------------------------------------------------
def compute_svm_loss(
    scores: np.ndarray,
    true_labels: np.ndarray,
    delta: float = 1.0,
) -> tuple[np.ndarray, float]:
    """PPT에서 배운 Multiclass SVM Hinge Loss를 계산합니다.

    이미지 i의 손실:
        L_i = sum_{j != y_i} max(0, s_j - s_y + delta)

    모든 예시의 평균 손실:
        L = (L_1 + ... + L_N) / N

    용어:
        s_y   : 정답 클래스의 점수
        s_j   : 비교 대상 클래스의 점수
        delta : 클래스 간 확보하려는 마진 (강의자료처럼 1 사용)

    중요한 점:
        1) 정답 클래스 자체는 합산에서 제외합니다.
        2) 음수 마진은 0으로 처리합니다.
        3) 정답 예측이라고 해서 반드시 손실이 0인 것은 아닙니다.
           정답 점수가 다른 점수보다 delta 이상 높아야 손실이 0입니다.
    """
    n = scores.shape[0]
    if n == 0 or scores.shape != (n, NUM_CLASSES):
        raise ValueError("scores는 (N, 10) 형태이며 N은 1 이상이어야 합니다.")
    if true_labels.shape != (n,):
        raise ValueError("true_labels는 (N,) 형태여야 합니다.")
    if delta <= 0:
        raise ValueError("delta는 0보다 커야 합니다.")

    # 각 이미지의 정답 클래스 점수를 추출합니다. (N,) -> (N, 1)
    correct_class_scores = scores[np.arange(n), true_labels][:, np.newaxis]

    # s_j - s_y + delta, 결과가 음수이면 0으로 바꿉니다.
    margins = np.maximum(0.0, scores - correct_class_scores + delta)

    # j == y_i인 정답 클래스 항은 수식에서 제외해야 합니다.
    # 이 작업을 하지 않으면 정답 클래스도 항상 delta만큼 손실에 더해집니다.
    margins[np.arange(n), true_labels] = 0.0

    losses_per_image = margins.sum(axis=1)
    average_loss = float(np.mean(losses_per_image))
    return losses_per_image, average_loss


def compute_softmax(scores: np.ndarray) -> np.ndarray:
    """점수(score)를 0~1 범위의 Softmax 확률로 변환합니다.

    수식: P(class=j | image=i) = exp(s_ij) / sum_k exp(s_ik)

    구현 방법:
        1) 각 행의 최대 점수를 빼줍니다.
        2) 지수함수 exp()를 적용합니다.
        3) 이미지별 지수 합계로 나눕니다.

    max를 빼는 이유:
        매우 큰 수의 exp 계산에서 overflow가 발생할 수 있기 때문입니다.
        모든 점수에 같은 값을 빼도 Softmax 확률은 바뀌지 않습니다.

    참고: 무작위 가중치의 Softmax 출력은 수학적으로 정규화된 값이지,
    학습된 모델처럼 신뢰할 수 있는 실제 정답 확률은 아닙니다.
    """
    if scores.ndim != 2 or scores.shape[1] != NUM_CLASSES:
        raise ValueError("scores는 (N, 10) 형태여야 합니다.")

    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probabilities


def verify_probabilities(probabilities: np.ndarray) -> tuple[np.ndarray, bool]:
    """각 이미지의 Softmax 확률 10개를 합쳐 1인지 검사합니다.

    실수의 부동소수점 오차 때문에 sum == 1 대신 np.allclose()를 씁니다.
    """
    sums = probabilities.sum(axis=1)
    return sums, bool(np.allclose(sums, 1.0, rtol=1e-10, atol=1e-10))


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
    """test_batch의 앞부분에서 맞은 사례와 틀린 사례를 각각 한 장 찾습니다.

    선택한 10장의 예시에서 우연히 맞은 결과가 없을 수도 있으므로,
    별도의 고정된 탐색 범위(기본 1000장)에서 맞은/틀린 예를 확인합니다.

    모델의 가중치는 바꾸지 않고, 정답 확인에만 실제 라벨을 사용합니다.
    탐색 과정은 학습도 최적화도 아닙니다.
    """
    count = min(max(1, search_size), len(all_images))
    X = preprocess_images(all_images[:count])
    predictions = predict_classes(compute_scores(X, W, b))
    matches = predictions == all_labels[:count]

    cases: dict[str, dict | None] = {"correct": None, "incorrect": None}
    for case_name, match_value in (("correct", True), ("incorrect", False)):
        positions = np.flatnonzero(matches == match_value)
        if len(positions) > 0:
            idx = int(positions[0])
            cases[case_name] = {
                "index": idx,
                "image": all_images[idx],
                "true_label": int(all_labels[idx]),
                "predicted_label": int(predictions[idx]),
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
    """10개 클래스의 가중치 벡터(3072)를 32x32x3 RGB 이미지로 보여줍니다.

    음수/양수 가중치는 원래 0~1 픽셀이 아니므로 시각화용으로 변환합니다.
    변환: 0 중심, +-3*표준편차가 대략 0~1에 대응하도록 조정.

    이 함수의 변환은 *그림을 보기 위한 것*이며 실제 W 값은 바뀌지 않습니다.
    """
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    visual_scale = 3.0 * float(np.std(W))
    for class_id, ax in enumerate(axes.flat):
        weight_image = flat_to_rgb_image(W[class_id])
        display_image = np.clip(weight_image / (2 * visual_scale) + 0.5, 0.0, 1.0)
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
def main() -> None:
    """실행 옵션을 읽고, 1~7단계를 순서대로 호출합니다."""
    parser = argparse.ArgumentParser(description="CIFAR-10 Linear Classification (no training)")
    parser.add_argument(
        "--data", type=Path, default=Path("cifar-10-python.tar.gz"),
        help="CIFAR-10 Python tar.gz 파일 또는 압축 해제된 폴더 경로",
    )
    parser.add_argument(
        "--download", action="store_true", help="데이터가 없다면 공식 사이트에서 다운로드",
    )
    parser.add_argument(
        "--num-examples", type=int, default=10,
        help="점수/확률/Loss를 계산할 예시 이미지 수 (기본: 10)",
    )
    parser.add_argument("--seed", type=int, default=42, help="재현용 난수 시드 (기본: 42)")
    parser.add_argument("--weight-std", type=float, default=0.01, help="무작위 가중치 표준편차")
    parser.add_argument("--search-size", type=int, default=1000, help="맞은/틀린 예시 탐색 범위")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="결과 저장 폴더")
    parser.add_argument("--self-test", action="store_true", help="PPT 수식만 검증 (데이터 필요 없음)")
    args = parser.parse_args()

    if args.self_test:
        print(run_formula_self_test())
        return

    # 1) 데이터셋을 불러옵니다. 필요한 경우에만 명시적으로 다운로드합니다.
    if args.download and not args.data.exists():
        if args.data.suffixes[-2:] != [".tar", ".gz"]:
            parser.error("--download 사용 시 --data에는 .tar.gz 파일 경로를 지정해 주세요.")
        download_cifar10(args.data)
    raw_images, labels = load_cifar10(args.data)
    print(f"[1] 데이터 로딩 완료: images={raw_images.shape}, labels={labels.shape}")

    # 2) 고정 시드로 이미지 예시를 선정합니다.
    selected_images, selected_labels, selected_indices = select_examples(
        raw_images, labels, args.num_examples, args.seed
    )
    X = preprocess_images(selected_images)
    print(f"[2] 예시 {len(X)}장, 정규화된 입력 X={X.shape}")

    # 3) 훈련하지 않은 무작위 파라미터로 점수를 계산합니다.
    W, b = initialize_parameters(args.seed, args.weight_std)
    scores = compute_scores(X, W, b)
    predictions = predict_classes(scores)
    print(f"[3] 가중치 W={W.shape}, 편향 b={b.shape}, 점수={scores.shape}")

    # 4) 정답을 이용해 SVM Hinge Loss만 계산합니다. W를 갱신하지 않습니다.
    losses_per_image, average_loss = compute_svm_loss(scores, selected_labels)
    print(f"[4] Multiclass SVM 평균 Hinge Loss: {average_loss:.6f}")

    # 5) 원본 점수를 Softmax 확률로 바꾸고 합이 1인지 확인합니다.
    probabilities = compute_softmax(scores)
    probability_sums, probability_valid = verify_probabilities(probabilities)
    if not probability_valid:
        raise AssertionError("Softmax 확률 합이 1이 아닙니다. 코드를 확인해 주세요.")
    if not np.array_equal(predictions, predict_classes(probabilities)):
        raise AssertionError("Softmax 처리로 예측 클래스가 달라졌습니다.")
    print(f"[5] Softmax 확률 합 검사: {probability_valid}")

    # 6) 별도 샘플 중 정답/오답 사례를 찾습니다(확률 높은 예시를 골라서 학습하지 않음).
    cases = find_correct_incorrect_examples(raw_images, labels, W, b, args.search_size)
    print("[6] 정답 및 오답 사례 탐색 완료")

    # 7) 모든 결과를 제출용 파일로 저장합니다.
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


if __name__ == "__main__":
    main()
