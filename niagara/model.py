from typing import List, Dict, Optional, Tuple
from .confidence_signals import ConfidenceSignal, ModelIntrinsicProb, ModelIntrinsicLogProb
from .confidence_calibrators import ConfidenceSignalCalibrator, NullCalibrator
from .confidence_transformations import ConfidenceSignalTransformation, NullTransformation
from .custom_types import ModelParameters
from .api_clients import APIClient, FireworksClient


class Model:
    def __init__(
        self,
        model_name: Optional[str] = None,
        thresholds: ModelParameters = None,
        conf_signal: ConfidenceSignal = ModelIntrinsicLogProb(),
        conf_signal_transform: ConfidenceSignalTransformation = NullTransformation(),
        conf_signal_calibrator: ConfidenceSignalCalibrator = NullCalibrator(),
        client: APIClient = FireworksClient(),
    ):
        """
        Initialize a model.
        """
        self.model_name = model_name
        self.thresholds = thresholds
        self.conf_signal = conf_signal
        self.conf_signal_transform = conf_signal_transform
        self.conf_signal_calibrator = conf_signal_calibrator
        self.client = client

        # figure out the cost per million tokens
        self.model_info = self.client.models.get(self.model_name)
        self.cpm_tokens = {"in": self.model_info['input_cost'], "out": self.model_info['output_cost'] }

    
    def __repr__(self) -> str:
        return (
            f"Model(model_name={self.model_name}, thresholds={self.thresholds}, " \
            f"conf_signal={self.conf_signal}, conf_signal_transform={self.conf_signal_transform}, "\
            f"conf_signal_calibrator={self.conf_signal_calibrator}, client={self.client})"
        )

    def query_raw_model(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        answer, logprobs, cost, latency, num_tokens = self.client.get_answer(
            model_name=self.model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return answer, logprobs, cost, latency, num_tokens

    def answer_query(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        prev_conf: Optional[List[float]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_not_calibrate: bool = False,
    ) -> Tuple[str, float, float]:
        """
        Return the answer, its confidence, and the cost.

        Returns:
        Tuple[str, float, float]: The answer, its calibrated confidence, and the cost.
        """
        if prev_conf is None:
            prev_conf = []

        answer, logprobs, cost, latency, num_tokens = self.query_raw_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        conf_signal, conf_cost, conf_latency = self.conf_signal.get_confidence_signal(
            model=self,
            answer=answer,
            logprobs=logprobs,
            prev_conf=prev_conf,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        transformed_conf_signal = self.conf_signal_transform.transform_confidence_signal(
            conf_signal
        )

        # Add in the cost of uncertainty quantification
        cost += conf_cost
        latency += conf_latency

        if do_not_calibrate:
            return answer, transformed_conf_signal, cost, latency, num_tokens
        else:
            calibrated_conf = self.conf_signal_calibrator.calibrate_confidence_signal(
                transformed_conf_signal
            )
            return answer, calibrated_conf, cost, latency, num_tokens
