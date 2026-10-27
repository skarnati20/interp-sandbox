from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import torch

from src.activations import ActivationExtractor, ExtractionResult
from src.probe.base import Probe


@dataclass
class SteeringDecision:
    """
    Encapsulates the action, raw extraction, probe evaluation, and intervention telemetry
    produced by a steering strategy during a single agent turn.
    """
    action_text: str
    result: ExtractionResult
    probe_score: float
    intervened: bool
    strategy_info: dict[str, Any] = field(default_factory=dict)


class BaseSteeringStrategy(ABC):
    """
    Abstract base class for steering strategies.
    Defines how probe predictions over residual-stream activations guide, modify,
    or resample model actions during rollouts.
    """

    def score_extraction(
        self,
        result: ExtractionResult,
        probe: Probe,
        layer: int,
        anchor: str,
    ) -> float:
        """
        Extracts activation vector at specified layer and anchor, then queries the probe
        to obtain predicted probability of success (class 1).
        """
        feat_tensor = result.get_anchor_state(anchor=anchor, layer_ids=[layer])
        feat_arr = feat_tensor.detach().cpu().numpy()
        if feat_arr.ndim == 1:
            feat_arr = np.expand_dims(feat_arr, axis=0)

        probs = probe.predict(feat_arr, return_proba=True)
        return float(probs[0])

    @abstractmethod
    def step(
        self,
        messages: list[dict],
        extractor: ActivationExtractor,
        probe: Probe,
        probe_layer: int,
        probe_anchor: str,
        temperature: float = 0.0,
    ) -> SteeringDecision:
        """
        Executes a turn step under this steering strategy, returning the chosen action,
        its extraction result, probe prediction score, and intervention telemetry.
        """
        pass


class PassiveSteeringStrategy(BaseSteeringStrategy):
    """
    Observational control strategy: generates actions normally without intervention,
    applying the probe purely as a passive monitor.
    """

    def step(
        self,
        messages: list[dict],
        extractor: ActivationExtractor,
        probe: Probe,
        probe_layer: int,
        probe_anchor: str,
        temperature: float = 0.0,
    ) -> SteeringDecision:
        result = extractor.step_forward(messages, temperature=temperature)
        action_text = result.completion_text.strip()
        score = self.score_extraction(result, probe, probe_layer, probe_anchor)

        return SteeringDecision(
            action_text=action_text,
            result=result,
            probe_score=score,
            intervened=False,
            strategy_info={"strategy": "passive", "attempts": 1},
        )


class ResamplingSteeringStrategy(BaseSteeringStrategy):
    """
    Rejection sampling / Best-of-N steering strategy.
    Generates candidate actions and queries the probe. If predicted success is below
    threshold, resamples stochastic actions until a candidate meets the threshold
    or max attempts is reached.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        max_attempts: int = 5,
        resample_temperature: float = 0.7,
        mode: str = "first_above_threshold",
    ):
        self.threshold = threshold
        self.max_attempts = max(1, max_attempts)
        self.resample_temperature = resample_temperature
        self.mode = mode

    def step(
        self,
        messages: list[dict],
        extractor: ActivationExtractor,
        probe: Probe,
        probe_layer: int,
        probe_anchor: str,
        temperature: float = 0.0,
    ) -> SteeringDecision:
        candidates: list[tuple[ExtractionResult, float, str]] = []

        for attempt in range(self.max_attempts):
            # First attempt can use user's baseline temp; subsequent resamples use resample_temperature
            curr_temp = temperature if attempt == 0 else max(self.resample_temperature, 0.1)
            result = extractor.step_forward(messages, temperature=curr_temp)
            action_text = result.completion_text.strip()
            score = self.score_extraction(result, probe, probe_layer, probe_anchor)
            candidates.append((result, score, action_text))

            if self.mode == "first_above_threshold" and score >= self.threshold:
                return SteeringDecision(
                    action_text=action_text,
                    result=result,
                    probe_score=score,
                    intervened=(attempt > 0),
                    strategy_info={
                        "strategy": "resampling",
                        "attempts": attempt + 1,
                        "met_threshold": True,
                        "scores": [c[1] for c in candidates],
                    },
                )

        # Select candidate with highest probe score
        best_result, best_score, best_action = max(candidates, key=lambda c: c[1])
        met_threshold = best_score >= self.threshold

        return SteeringDecision(
            action_text=best_action,
            result=best_result,
            probe_score=best_score,
            intervened=(len(candidates) > 1),
            strategy_info={
                "strategy": "resampling",
                "attempts": len(candidates),
                "met_threshold": met_threshold,
                "scores": [c[1] for c in candidates],
            },
        )


class VectorSteeringStrategy(BaseSteeringStrategy):
    """
    Representation engineering / Activation addition steering strategy.
    Injects a steering direction vector into residual-stream activations at a target layer.
    Vector can be explicitly provided or automatically derived from a trained LinearProbe.
    """

    def __init__(
        self,
        steering_vector: Optional[np.ndarray | torch.Tensor] = None,
        coeff: float = 1.0,
        condition: str = "threshold_gated",
        threshold: float = 0.5,
        injection_layer: Optional[int] = None,
    ):
        self.steering_vector = steering_vector
        self.coeff = coeff
        self.condition = condition  # "threshold_gated" or "always"
        self.threshold = threshold
        self.injection_layer = injection_layer

    def resolve_vector(self, probe: Probe) -> np.ndarray:
        if self.steering_vector is not None:
            if isinstance(self.steering_vector, torch.Tensor):
                return self.steering_vector.detach().cpu().numpy()
            return np.asarray(self.steering_vector)

        if hasattr(probe, "steering_vector"):
            return probe.steering_vector

        raise ValueError(
            "VectorSteeringStrategy requires an explicit steering_vector or a probe "
            "with a 'steering_vector' property (e.g. LinearProbe)."
        )

    def step(
        self,
        messages: list[dict],
        extractor: ActivationExtractor,
        probe: Probe,
        probe_layer: int,
        probe_anchor: str,
        temperature: float = 0.0,
    ) -> SteeringDecision:
        layer_to_steer = self.injection_layer if self.injection_layer is not None else probe_layer
        vec = self.resolve_vector(probe)

        if self.condition == "always":
            with extractor.steer(layer_idx=layer_to_steer, vector=vec, coeff=self.coeff):
                steered_result = extractor.step_forward(messages, temperature=temperature)
            action_text = steered_result.completion_text.strip()
            score = self.score_extraction(steered_result, probe, probe_layer, probe_anchor)

            return SteeringDecision(
                action_text=action_text,
                result=steered_result,
                probe_score=score,
                intervened=True,
                strategy_info={
                    "strategy": "vector_steering",
                    "condition": "always",
                    "coeff": self.coeff,
                    "layer": layer_to_steer,
                },
            )

        # "threshold_gated"
        baseline_result = extractor.step_forward(messages, temperature=temperature)
        baseline_score = self.score_extraction(baseline_result, probe, probe_layer, probe_anchor)

        if baseline_score >= self.threshold:
            return SteeringDecision(
                action_text=baseline_result.completion_text.strip(),
                result=baseline_result,
                probe_score=baseline_score,
                intervened=False,
                strategy_info={
                    "strategy": "vector_steering",
                    "condition": "threshold_gated",
                    "intervened": False,
                    "baseline_score": baseline_score,
                },
            )

        # Baseline failed threshold -> apply steering vector
        with extractor.steer(layer_idx=layer_to_steer, vector=vec, coeff=self.coeff):
            steered_result = extractor.step_forward(messages, temperature=temperature)
        action_text = steered_result.completion_text.strip()
        steered_score = self.score_extraction(steered_result, probe, probe_layer, probe_anchor)

        return SteeringDecision(
            action_text=action_text,
            result=steered_result,
            probe_score=steered_score,
            intervened=True,
            strategy_info={
                "strategy": "vector_steering",
                "condition": "threshold_gated",
                "intervened": True,
                "baseline_score": baseline_score,
                "steered_score": steered_score,
                "coeff": self.coeff,
                "layer": layer_to_steer,
            },
        )
