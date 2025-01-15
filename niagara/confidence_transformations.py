import numpy as np
from abc import ABC

class ConfidenceSignalTransformation(ABC):
    """Abstract base class for confidence signal transformations. """
    
    def transform_confidence_signal(
        self, conf_signal: float | np.ndarray
    ) -> float | np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__


class NullTransformation(ConfidenceSignalTransformation):
    """ Class for leaving raw confidence signal untransformed. """

    def transform_confidence_signal(self, conf_signal):
        return conf_signal
    

class OneSidedAsymptoticLog(ConfidenceSignalTransformation):
    """ Class for applying the one-sided asymptotic transformation. """

    def __init__(self, drop_perfect_conf = False):
        self.drop_perfect_conf = drop_perfect_conf

    @staticmethod
    def safe_onesided_asymptotic_transform(logprobs, drop_zeros=True):
        logprobs = np.array(logprobs)

        if drop_zeros: logprobs = logprobs[logprobs != 0]

        perfect_conf = (logprobs == 0.0)
        use_large_p_approx = (logprobs > -1e-5) & (logprobs < 0.0)
        use_small_p_approx = (logprobs < -10)
        no_approx_needed = (logprobs >= -10) & (logprobs <= -1e-5)

        output = np.empty_like(logprobs)
        output[perfect_conf] = np.inf
        output[use_large_p_approx] = -np.log(-logprobs[use_large_p_approx])
        output[use_small_p_approx] = np.exp(logprobs[use_small_p_approx])
        output[no_approx_needed] = -np.log(1-np.exp(logprobs[no_approx_needed]))

        return output

    def transform_confidence_signal(self, conf_signal):
        if isinstance(conf_signal, float):
            conf_signal = [conf_signal]
            return_float = True
        else:
            return_float = False
        
        conf_signal = np.array(conf_signal)
        transformed_confidence = OneSidedAsymptoticLog.safe_onesided_asymptotic_transform(
            logprobs=conf_signal,
            drop_zeros=self.drop_perfect_conf
        )

        return transformed_confidence if not return_float else transformed_confidence[0]


class TwoSidedAsymptoticLog(ConfidenceSignalTransformation):
    """ Class for applying the one-sided asymptotic transformation. """

    def __init__(self, drop_perfect_conf = False):
        self.drop_perfect_conf = drop_perfect_conf

    def transform_confidence_signal(self, conf_signal):
        if isinstance(conf_signal, float):
            conf_signal = [conf_signal]
            return_float = True
        else:
            return_float = False

        conf_signal = np.array(conf_signal)
        transformed_confidence = np.empty_like(conf_signal)

        # use safe_onesided_asymptotic_transform for log p > log(0.5)
        logprob_more_than_half = (conf_signal >= np.log(0.5))
        transformed_confidence[logprob_more_than_half] = -np.log(2) + OneSidedAsymptoticLog.safe_onesided_asymptotic_transform(
            conf_signal[logprob_more_than_half],
            drop_zeros=False
        )

        # for p < 0.5, we use log(2) - log(1/p) = log(2) + log p
        logprob_less_than_half = (conf_signal < np.log(0.5))
        transformed_confidence[logprob_less_than_half] = np.log(2) + conf_signal[logprob_less_than_half]

        if self.drop_perfect_conf:
            return transformed_confidence[~np.isinf(transformed_confidence)] if not return_float else transformed_confidence[~np.isinf(transformed_confidence)][0]
        else:
            return transformed_confidence  if not return_float else transformed_confidence[0]