"""과제 구현 자체의 간단한 단위 테스트. CIFAR-10 원본 파일 없이 실행 가능합니다."""
import unittest
import numpy as np
from linear_classification import (
    INPUT_DIM, NUM_CLASSES, initialize_parameters, compute_scores,
    compute_svm_loss, compute_softmax, verify_probabilities,
    flat_to_rgb_image, select_examples,
)


class TestLinearClassification(unittest.TestCase):
    def test_reproducible_parameters(self):
        w1, b1 = initialize_parameters(seed=42)
        w2, b2 = initialize_parameters(seed=42)
        self.assertEqual(w1.shape, (NUM_CLASSES, INPUT_DIM))
        self.assertEqual(b1.shape, (NUM_CLASSES,))
        np.testing.assert_array_equal(w1, w2)
        np.testing.assert_array_equal(b1, b2)
        self.assertTrue(np.all(b1 == 0))

    def test_score_shapes_and_formula(self):
        X = np.zeros((2, INPUT_DIM))
        X[0, 0] = 1.0
        W = np.zeros((NUM_CLASSES, INPUT_DIM))
        W[3, 0] = 2.5
        b = np.zeros(NUM_CLASSES)
        scores = compute_scores(X, W, b)
        self.assertEqual(scores.shape, (2, NUM_CLASSES))
        self.assertEqual(scores[0, 3], 2.5)
        self.assertEqual(scores[1, 3], 0.0)

    def test_svm_loss_ppt(self):
        # PPT 23~25페이지 예시: 나머지 클래스 점수는 매우 낮게 설정합니다.
        s = np.full((3, NUM_CLASSES), -100.0)
        s[:, :3] = [[3.2, 5.1, -1.7], [1.3, 4.9, 2.0], [2.2, 2.5, -3.1]]
        losses, avg_loss = compute_svm_loss(s, np.array([0, 1, 2]))
        np.testing.assert_allclose(losses, [2.9, 0.0, 12.9])
        self.assertAlmostEqual(avg_loss, 15.8 / 3)

    def test_softmax_sums_to_one_and_stability(self):
        s = np.array([[1000.0] + [999.0] * 9, [-1000.0] * 10])
        p = compute_softmax(s)
        sums, valid = verify_probabilities(p)
        self.assertTrue(valid)
        np.testing.assert_allclose(sums, [1.0, 1.0])
        np.testing.assert_array_equal(np.argmax(s, axis=1), np.argmax(p, axis=1))

    def test_cifar_image_channel_layout(self):
        flat = np.zeros(INPUT_DIM, dtype=np.uint8)
        flat[0] = 255            # R 채널의 첫 픽셀
        flat[1024] = 128         # G 채널의 첫 픽셀
        flat[2048] = 64          # B 채널의 첫 픽셀
        rgb = flat_to_rgb_image(flat)
        self.assertEqual(rgb.shape, (32, 32, 3))
        np.testing.assert_array_equal(rgb[0, 0], [255, 128, 64])

    def test_balanced_example_selection(self):
        # 실제 CIFAR 파일이 없어도 샘플 선택이 각 클래스를 고르는지 테스트합니다.
        images = np.zeros((100, INPUT_DIM), dtype=np.uint8)
        labels = np.arange(100) % 10
        chosen, y, ids = select_examples(images, labels, 10, seed=42)
        self.assertEqual(chosen.shape, (10, INPUT_DIM))
        np.testing.assert_array_equal(y, np.arange(10))
        self.assertEqual(len(np.unique(ids)), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
