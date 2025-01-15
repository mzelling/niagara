import numpy as np
import pandas as pd
from .custom_types import ModelCalibrationData, AllModelCalibrationData
import statsmodels.api as sm
from statsmodels.formula.api import logit
from scipy.optimize import minimize
from abc import ABC

class NotCalibratedError(Exception):
    """
    Throw this exception when a ConfidenceSignalCalibrator
    is called even though it has not been calibrated.
    """

    def __init__(self, message):
        super().__init__(message)


class ConfidenceSignalCalibrator(ABC):
    """Abstract class for confidence signal calibrators."""

    def calibrate_confidence_signal(
        self, transformed_conf_signal: float | list[float] | np.ndarray
    ) -> float | list[float] | np.ndarray:
        """Calibrate the confidence signal."""
        pass

    def calibrate(self, data: ModelCalibrationData) -> None:
        """Calibrate the model using provided data."""
        pass

    def __repr__(self):
        return self.__class__.__name__


class NullCalibrator(ConfidenceSignalCalibrator):
    """A calibrator that returns the confidence signal unchanged."""

    def calibrate_confidence_signal(
        self, transformed_conf_signal: float | list[float] | np.ndarray
    ) -> np.ndarray:
        """Return the input confidence signal without changes."""
        return transformed_conf_signal

    def calibrate(self, data: ModelCalibrationData) -> None:
        """No calibration needed for null calibrator."""
        pass


class LogisticRegressionCalibrator(ConfidenceSignalCalibrator):
    """
    Calibrator that calibrates the transformed confidence signal 
    with logistic regression.
    """

    def __init__(self, thold=1e-16, drop_perfect_conf=False):
        self.logreg = None
        self.thold = thold
        self.drop_perfect_conf = drop_perfect_conf
        self.FINITE_VALUE_FOR_POS_INF_CONFIDENCE = None
        self.FINITE_VALUE_FOR_NEG_INF_CONFIDENCE = None

    def calibrate_confidence_signal(
        self, transformed_conf_signal: float | list[float] | np.ndarray
    ) -> np.ndarray:
        """ Apply calibrated LogisticRegressionCalibrator to calibrate transformed confidence values. """
        if self.logreg is None:
            raise NotCalibratedError(
                "LogisticRegressionCalibrator requires calibration; call its .calibrate(...) method."
            )
        
        if isinstance(transformed_conf_signal, float):
            is_float = True
            transformed_conf_signal = [transformed_conf_signal]
        else:
            is_float = False

        transformed_conf_signal = np.array(transformed_conf_signal)

        # set infinite values to the maximum of finite ones
        if not self.drop_perfect_conf:
            pos_inf_confidence = np.isposinf(transformed_conf_signal)
            neg_inf_confidence = np.isneginf(transformed_conf_signal)
            transformed_conf_signal[pos_inf_confidence] = self.FINITE_VALUE_FOR_POS_INF_CONFIDENCE
            transformed_conf_signal[neg_inf_confidence] = self.FINITE_VALUE_FOR_NEG_INF_CONFIDENCE
        else:
            transformed_conf_signal = transformed_conf_signal[~np.isinf(transformed_conf_signal)]

        calibrated_conf_signal = self.logreg.predict(
            sm.add_constant(transformed_conf_signal, has_constant='add')
        )
        return calibrated_conf_signal if not is_float else calibrated_conf_signal[0]

    def calibrate(self, data: ModelCalibrationData):
        """Fit logistic regression to data."""
        transformed_conf_signal = np.array(data["transformed_confidence"])
        correctness = np.array(data["correctness"])

        pos_inf_confidence = np.isposinf(transformed_conf_signal)
        neg_inf_confidence = np.isneginf(transformed_conf_signal)

        if not self.drop_perfect_conf:
            # replace +inf with maximum finite value
            max_finite_conf = np.max(transformed_conf_signal[~pos_inf_confidence])
            transformed_conf_signal[pos_inf_confidence] = max_finite_conf
            self.FINITE_VALUE_FOR_POS_INF_CONFIDENCE = max_finite_conf
            # replace -inf with minimum finite value
            min_finite_conf = np.min(transformed_conf_signal[~neg_inf_confidence])
            transformed_conf_signal[neg_inf_confidence] = min_finite_conf
            self.FINITE_VALUE_FOR_NEG_INF_CONFIDENCE = min_finite_conf
        else:
            infinite_confidence = np.isinf(transformed_conf_signal)
            transformed_conf_signal = transformed_conf_signal[~infinite_confidence]
            correctness = correctness[~infinite_confidence]

        df = pd.DataFrame(
            {
                "y": correctness.astype(int),
                "x": transformed_conf_signal,
            }
        )

        model = sm.Logit(df["y"], sm.add_constant(df["x"], has_constant='add'))
        fitted_model = model.fit(maxiter=1000)
        self.logreg = fitted_model


class AllModelCorrectnessPredictor(ConfidenceSignalCalibrator):
    def __init__(self, drop_perfect_conf=False, alpha=0.1):
        self.logistic_regressions = []
        self.alpha = alpha
        self.drop_perfect_conf = drop_perfect_conf
        self.FINITE_VALUE_FOR_POS_INF_CONFIDENCE = None
        self.FINITE_VALUE_FOR_NEG_INF_CONFIDENCE = None

    def calibrate_confidence_signal(
        self, transformed_conf_signal: float | list[float] | np.ndarray,
        lower_quantile=0.05, upper_quantile=0.95
    ) -> np.ndarray:
        """ Predict correctness of all models using the transformed confidence signals. """
        if len(self.logistic_regressions) == 0:
            raise NotCalibratedError(
                f"{self.__class__.__name__} requires calibration; call its .calibrate(...) method."
            )
    
        if isinstance(transformed_conf_signal, float):
            is_float = True
            transformed_conf_signal = [transformed_conf_signal]
        else:
            is_float = False

        transformed_conf_signal = np.array(transformed_conf_signal)

        # set infinite values to the maximum of finite ones
        if not self.drop_perfect_conf:
            # set finite value for +inf
            pos_inf_confidence = np.isposinf(transformed_conf_signal)
            transformed_conf_signal[pos_inf_confidence] = self.FINITE_VALUE_FOR_POS_INF_CONFIDENCE
            # set finite value for -inf
            neg_inf_confidence = np.isneginf(transformed_conf_signal)
            transformed_conf_signal[neg_inf_confidence] = self.FINITE_VALUE_FOR_NEG_INF_CONFIDENCE
        else:
            transformed_conf_signal = transformed_conf_signal[~np.isinf(transformed_conf_signal)]

        input_for_logreg = sm.add_constant(transformed_conf_signal, has_constant='add')

        predictions = [
            logreg.get_prediction(input_for_logreg) for logreg in self.logistic_regressions
        ]

        predicted_means = [ prdxn.predicted for prdxn in predictions ]

        conf_ints = [
            prdxn.conf_int(alpha=self.alpha) for prdxn in predictions
        ]

        return {
            'correctness': predicted_means if not is_float else [
                corr_prdxn[0] for corr_prdxn in predicted_means
            ],
            'confidence_intervals': conf_ints if not is_float else [
                conf_int[0] for conf_int in conf_ints
            ]
        }
    
    def calibrate(self, data: AllModelCalibrationData):
        """Fit logistic regression to data."""
        transformed_conf_signal = np.array(data["transformed_confidence"])

        if not self.drop_perfect_conf:
            # set finite max value for +inf
            pos_inf_confidence = np.isposinf(transformed_conf_signal)
            max_finite_conf = np.max(transformed_conf_signal[~pos_inf_confidence])
            transformed_conf_signal[pos_inf_confidence] = max_finite_conf
            self.FINITE_VALUE_FOR_POS_INF_CONFIDENCE = max_finite_conf
            # set finite min value for -inf
            neg_inf_confidence = np.isneginf(transformed_conf_signal)
            min_finite_conf = np.min(transformed_conf_signal[~neg_inf_confidence])
            transformed_conf_signal[neg_inf_confidence] = min_finite_conf
            self.FINITE_VALUE_FOR_NEG_INF_CONFIDENCE = min_finite_conf
        else:
            infinite_confidence = np.isinf(transformed_conf_signal)
            transformed_conf_signal = transformed_conf_signal[~infinite_confidence]
            correctness = correctness[~infinite_confidence]

        for i, model_correctness in enumerate(data["correctness"]):
            correctness = np.array(model_correctness)

            df = pd.DataFrame(
                {
                    "y": correctness.astype(int),
                    "x": transformed_conf_signal,
                }
            )

            model = sm.Logit(df["y"], sm.add_constant(df["x"], has_constant='add'))
            fitted_model = model.fit(maxiter=1000)
            self.logistic_regressions.append(fitted_model)



class TemperatureScalingCalibrator(ConfidenceSignalCalibrator):
    """Calibrator that scales the confidence signal by a temperature parameter."""

    DEFAULT_THRESHOLD = 1e-16
    MAX_ITERATIONS = 1000

    def __init__(self, loss = "ECE"):
        self.temperature = None
        self.loss = loss

    def calibrate_confidence_signal(
        self,
        transformed_conf_signal: float | list[float] | np.ndarray,
        thold: float = DEFAULT_THRESHOLD,
    ) -> np.ndarray:
        """Apply temperature scaling to the confidence signal."""
        if self.temperature is None:
            raise NotCalibratedError(
                "TemperatureScalingCalibrator requires calibration; call its .calibrate(...) method."
            )

        is_float = isinstance(transformed_conf_signal, float)
        conf_signal = np.array(conf_signal if not is_float else [conf_signal])
        transformed_conf_signal = np.log(
            np.clip(conf_signal, thold, 1.0) / np.clip(1 - conf_signal, thold, 1.0)
        )
        scaled_conf_signal = np.clip(transformed_conf_signal / self.temperature, -100, 100)
        calibrated_conf_signal = np.where(
            scaled_conf_signal >= 0,
            1 / (1 + np.exp(-scaled_conf_signal)),
            np.exp(scaled_conf_signal) / (1 + np.exp(scaled_conf_signal))
        )
        return calibrated_conf_signal if not is_float else calibrated_conf_signal[0]

    def calibrate(
        self,
        data: ModelCalibrationData,
        thold: float = DEFAULT_THRESHOLD,
        max_iter: int = MAX_ITERATIONS,
        loss: str | None = None,
        num_bins: int = 15,
    ):
        """Fit temperature parameter using provided data.

        Parameters:
            data (ModelCalibrationData): The data to use for calibration.
            thold (float): Threshold to avoid division by zero.
            max_iter (int): Maximum number of iterations for the optimizer.
            loss (str): Loss function to use for optimization ('NLL' or 'ECE').
            num_bins (int): Number of bins to use when computing ECE.
        """
        loss = self.loss if loss is None else loss
        
        confidence = np.array(data["transformed_confidence"])
        correctness = np.array(data["correctness"]).astype(int)

        if len(confidence) != len(correctness):
            raise ValueError(
                "Confidence and correctness arrays must have the same length"
            )
        if not (0 <= confidence).all() and (confidence <= 1).all():
            raise ValueError(
                "For temperature scaling, transformed confidence values must be between 0 and 1"
            )

        logits = np.log(
            np.clip(confidence, thold, 1.0) / np.clip(1 - confidence, thold, 1.0)
        )

        if loss == "NLL":
            def nll_loss(T):
                scaled_logits = logits / T[0]
                log_prob = -np.log(1 + np.exp(-scaled_logits))
                loss_value = -np.mean(
                    correctness * log_prob
                    + (1 - correctness) * (log_prob - scaled_logits)
                )
                return loss_value

            initial_T = [1.0]
            bounds = [(thold, None)]
            res = minimize(
                nll_loss,
                initial_T,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": max_iter},
            )
        elif loss == 'ECE':
            def ece_loss(T):
                scaled_logits = logits / T[0]
                calibrated_probs = 1 / (1 + np.exp(-scaled_logits))
        
                bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]

                ece = 0.0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (calibrated_probs >= bin_lower) & (calibrated_probs < bin_upper)
                    prop_in_bin = np.mean(in_bin)
                    if prop_in_bin > 0:
                        acc_in_bin = np.mean(correctness[in_bin])
                        avg_conf_in_bin = np.mean(calibrated_probs[in_bin])
                        ece += prop_in_bin * np.abs(acc_in_bin - avg_conf_in_bin)
                return ece

            initial_T = [1.0]
            bounds = [(thold, None)]
            res = minimize(
                ece_loss,
                initial_T,
                bounds=bounds,
                method="Nelder-Mead", 
                options={"maxiter": max_iter},
            )

        else:
            raise ValueError("Invalid loss function. Choose 'NLL' or 'ECE'.")

        if not res.success:
            raise RuntimeError(
                f"Temperature calibration failed to converge: {res.message}"
            )

        self.temperature = res.x[0]


if __name__ == "__main__":
    model_calibration_data: ModelCalibrationData = {
        "transformed_confidence": [0.1, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0],
        "correctness": [False, False, True, False, True, False, True, True],
    }
    # plattscaling_onesided = PlattScalingWithOneSidedTransform()
    # plattscaling_onesided.calibrate(model_calibration_data)
    # raw_conf_onesided = [0.0, 0.5, 0.75, 1.0]
    # calibrated_conf = plattscaling_onesided.calibrate_confidence_signal(
    #     raw_conf_onesided
    # )
    # print(f"Calibrated confidences: {calibrated_conf}")

    # plattscaling_twosided = PlattScalingWithTwoSidedTransform()
    # plattscaling_twosided.calibrate(model_calibration_data)
    # raw_conf_twosided = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    # calibrated_conf = plattscaling_twosided.calibrate_confidence_signal(
    #     raw_conf_twosided
    # )
    # print(f"Calibrated confidences: {calibrated_conf}")

    temp_scaling_calibrator = TemperatureScalingCalibrator()
    # temp_scaling_calibrator.calibrate(model_calibration_data, loss="NLL")
    temp_scaling_calibrator.calibrate(model_calibration_data, loss="ECE", num_bins=10)
    raw_conf_temp_scaling = [0.0, 0.5, 0.75, 1.0]
    calibrated_conf = temp_scaling_calibrator.calibrate_confidence_signal(
        raw_conf_temp_scaling
    )
    print(f"Temperature: {temp_scaling_calibrator.temperature}")
    print(f"Calibrated confidences: {calibrated_conf}")
