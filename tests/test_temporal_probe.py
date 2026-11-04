import unittest

import numpy as np
import pandas as pd

from src.store import ActivationDataset
from src.probe.probes.temporal_probe import TemporalProbe
from src.probe.train import train_and_evaluate


def dataset():
    metadata = pd.DataFrame([
        {"task_id": f"task-{episode}", "episode_id": str(episode),
         "round": turn, "anchor": "pre_gen", "success": episode % 2}
        for episode in range(8) for turn in range(3)
    ])
    x = np.array([[episode * 10 + turn, (episode % 2) * 5 + turn]
                  for episode in range(8) for turn in range(3)], dtype=float)
    return ActivationDataset(x, metadata.success.to_numpy(), [18], metadata)


class TestTemporalProbe(unittest.TestCase):
    def test_features_and_episode_reset(self):
        data = dataset()
        features = TemporalProbe().temporal_features(data)
        np.testing.assert_allclose(features[2], [2, 2, 1, 1, 1, 1])
        np.testing.assert_allclose(features[3], [10, 5, 10, 5, 0, 0])

    def test_shuffled_rows_restore_original_alignment(self):
        data = dataset()
        order = np.random.default_rng(42).permutation(len(data.y))
        probe = TemporalProbe()
        np.testing.assert_allclose(probe.temporal_features(data.slice(order)),
                                   probe.temporal_features(data)[order])

    def test_future_and_labels_do_not_affect_earlier_features(self):
        data = dataset()
        probe = TemporalProbe()
        expected = probe.temporal_features(data)
        data.X[2] = 1e6
        data.y = 1 - data.y
        data.metadata["success"] = data.y
        actual = probe.temporal_features(data)
        np.testing.assert_allclose(actual[:2], expected[:2])
        np.testing.assert_allclose(actual[3:], expected[3:])

    def test_anchor_streams_are_independent(self):
        data = dataset()
        other = data.metadata.assign(anchor="post_gen")
        mixed = ActivationDataset(np.concatenate([data.X, data.X + 100]),
                                  np.tile(data.y, 2), [18],
                                  pd.concat([data.metadata, other], ignore_index=True))
        features = TemporalProbe().temporal_features(mixed)
        np.testing.assert_allclose(features[24], [100, 100, 100, 100, 0, 0])

    def test_invalid_histories_fail(self):
        for rows in ([1, 2], [0, 2], [0, 0]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "contiguous prefix"):
                TemporalProbe().temporal_features(dataset().slice(np.array(rows)))
        data = dataset()
        data.metadata = data.metadata.drop(columns="episode_id")
        with self.assertRaisesRegex(ValueError, "episode_id"):
            TemporalProbe().temporal_features(data)
        with self.assertRaises(TypeError):
            TemporalProbe().temporal_features(np.zeros((3, 2)))

    def test_shared_pipeline_and_prefix_prediction(self):
        data = dataset()
        train = data.slice(np.arange(18))
        test = data.slice(np.arange(18, 24))
        probe = TemporalProbe(c_reg=0.1)
        result = train_and_evaluate(probe, train_set=train, eval_set=test,
                                    train_max_turn=2, eval_max_turn=3)
        self.assertEqual(result["train_samples"], 12)
        self.assertEqual(result["eval_samples"], 6)
        before = probe.scaler.mean_.copy()
        scores = probe.predict(test, return_proba=True)
        np.testing.assert_allclose(probe.predict(test.slice(np.array([0, 1])), return_proba=True),
                                   scores[:2])
        np.testing.assert_array_equal(probe.predict(test), scores >= 0.5)
        np.testing.assert_array_equal(probe.scaler.mean_, before)
        self.assertTrue(np.isfinite(result["eval_metrics"]["roc_auc"]))

    def test_step_idx_fallback(self):
        data = dataset()
        expected = TemporalProbe().temporal_features(data)
        data.metadata = data.metadata.rename(columns={"round": "step_idx"})
        np.testing.assert_allclose(TemporalProbe().temporal_features(data), expected)


if __name__ == "__main__":
    unittest.main()
