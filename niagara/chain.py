from typing import List, Optional, Literal
from .model import Model
from .exceptions import LLMUncertaintyError, LLMBadOutputError
from .api_clients import FireworksClient, OpenAIClient
from .custom_types import ModelParameters, ModelPerformanceData, ModelCalibrationData, AllModelCalibrationData
from .confidence_calibrators import AllModelCorrectnessPredictor
from .utils import compute_ece
import numpy as np
from tqdm import tqdm


class ChainResponse:
    def __init__(
        self,
        # answer: Optional[str] = None,
        # conf: Optional[float] = None,
        # latency: Optional[float] = None,
        # cost: Optional[float] = None,
        # final_model_idx: Optional[int] = None,
        # final_model_name: Optional[str] = None,
        # all_answers: Optional[List[str]] = None,
        # all_confidences: Optional[List[float]] = None,
        # all_model_names: Optional[List[str]] = None,
        # all_costs: Optional[List[float]] = None,
        # all_latencies: Optional[List[float]] = None,
        # model_trace: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Initialize a response from a chain of models.
        """
        self.attributes = []

        for k, v in kwargs.items():
            self.__setattr__(k, v)
            self.attributes.append(k)

        # self.answer = answer
        # self.confidence = conf
        # self.cost = cost
        # self.latency = latency
        # self.final_model_idx = final_model_idx
        # self.final_model_name = final_model_name
        # self.all_answers = all_answers or []
        # self.all_confidences = all_confidences or []
        # self.all_model_names = all_model_names or []
        # self.all_costs = all_costs or []
        # self.all_latencies = all_latencies or []

    def __repr__(self) -> str:
        attr_to_print = [ (k, self.__getattribute__(k)) for k in self.attributes ]

        start = "ChainResponse(\n\t"
        middle = ",\n\t".join([ f"{k}={v}" for (k, v) in attr_to_print ])
        end = "\n)"
        
        # return (
        #     f"ChainResponse(\n\tanswer={self.answer},\n\tconfidence={self.confidence},\n\tcost={self.cost}, "
        #     f"\n\tlatency={self.latency},\n\tfinal_model_index={self.final_model_idx}, "
        #     f"\n\tfinal_model_name={self.final_model_name},\n\tall_answers={self.all_answers}, "
        #     f"\n\tall_confidences={self.all_confidences},\n\tall_model_names={self.all_model_names}, "
        #     f"\n\tall_costs={self.all_costs},\n\tall_latencies={self.all_latencies}\n)"
        # )

        return start + middle + end
    

class Chain:

    def __init__(
        self,
        models: List[Model],
        allowed_outputs: Optional[List[str]] = None,
        rejection_symbol="[REJECT]",
        max_new_tokens=None,
    ):
        """
        Initialize a chain of models to process queries.
        """
        if not models:
            raise ValueError("The models list cannot be empty.")
        self.models = models
        self.model_names = [model.model_name for model in models]
        self.allowed_outputs = allowed_outputs
        self.max_new_tokens = max_new_tokens
        if (self.allowed_outputs is not None) and (max_new_tokens is None):
            self.max_new_tokens = max([len(x) for x in self.allowed_outputs])
        self.REJECT_QUERY_ANSWER = rejection_symbol

    def simulate_answer_query(
        self,
        model_params: list[ModelParameters] = None,
        chain_response_orig: ChainResponse = None,
        transform=False,
        calibrate=False
    ):
        """
        Re-run the chain logic for a given ChainResponse, with different
        model parameters.

        The chain can be a subset of the original chain.
        """
        assert len(model_params) == len(self.models)
        # assert len(chain_response.all_answers) == len(self.models)
        # assert len(chain_response.all_confidences) == len(self.models)

        final_answer = None
        final_model_idx = None

        # map old names to new names

        names_orig = chain_response_orig.all_model_names
        names = self.model_names

        mapping = [-1] * len(names)
        inverse_mapping = [-1] * len(names_orig)
        for i, name in enumerate(names):
            match = [ j for j, name_orig in enumerate(names_orig) if name == name_orig ]
            assert len(match) == 1
            j = match[0]
            mapping[i] = j
            inverse_mapping[j] = i

        def map_idx_to_orig(m_idx):
            """ Return model index corresponding to this model in the original chain. """
            return mapping[m_idx]

        for model_idx, model in enumerate(self.models):
            # Get confidence and transform/calibrate if needed
            conf = chain_response_orig.all_confidences[map_idx_to_orig(model_idx)]
            if transform:
                conf = self.models[model_idx].conf_signal_transform.transform_confidence_signal(conf)
            if calibrate:
                conf = self.models[model_idx].conf_signal_calibrator.calibrate_confidence_signal(conf)
            
            # Run the logic
            if (conf < model_params[model_idx]["reject"]):
                final_answer = self.REJECT_QUERY_ANSWER
                final_model_idx = model_idx
                break
            elif (conf >= model_params[model_idx]["accept"]):
                final_answer = chain_response_orig.all_answers[map_idx_to_orig(model_idx)]
                final_model_idx = model_idx
                break
            elif model_idx == len(self.models) - 1:
                final_answer = chain_response_orig.all_answers[map_idx_to_orig(model_idx)]
                final_model_idx = model_idx

        final_model_idx_orig = map_idx_to_orig(final_model_idx)
        new_costs = [
            chain_response_orig.all_costs[map_idx_to_orig(j)] for j in range(final_model_idx+1)
        ]
        new_latencies = [
            chain_response_orig.all_latencies[map_idx_to_orig(j)] for j in range(final_model_idx+1)
        ]
        all_new_tokens = [
            chain_response_orig.all_tokens[map_idx_to_orig(j)] for j in range(len(self.models))
        ]

        return ChainResponse(
            answer=final_answer,
            conf=conf,
            cost=sum(new_costs[:(final_model_idx+1)]),
            latency=sum(new_latencies[:(final_model_idx+1)]),
            all_tokens=all_new_tokens,
            final_model_idx=final_model_idx,
            final_model_name=self.model_names[final_model_idx],
            # conf=chain_response_orig.all_confidences[map_idx_to_orig(final_model_idx)],
            # cost=sum(chain_response_orig.all_costs[:(final_model_idx_orig+1)]),
            # latency=sum(chain_response.all_latencies[:(final_model_idx_orig+1)]),
            # all_answers=grab_subset(chain_response_orig.all_answers),
            # all_confidences=grab_subset(chain_response_orig.all_confidences),
            # all_model_names=grab_subset(chain_response_orig.all_model_names),
            # all_costs=chain_response_orig.all_costs,
            # all_latencies=chain_response_orig.all_latencies,
            # all_tokens=chain_response_orig.all_tokens
        )
    
    def evaluate_on_ground_truth(
            self, system_prompts, user_prompts, ground_truth_answers,
            metric="em", do_not_calibrate=True
        ):
        model_data = [{"transformed_confidence": [], "correctness": []} for model in self.models]
        final_model_data = []

        for system_prompt, user_prompt, gt_answer in tqdm(zip(
                system_prompts, user_prompts, ground_truth_answers
            )):
            response = self.answer_query(
                system_prompt=system_prompt, 
                user_prompt=user_prompt, 
                run_all_models=True,
                do_not_calibrate=do_not_calibrate
            )

            final_model_data.append(response.final_model_idx)

            if metric=="em":
                for i, model in enumerate(self.models):
                    model_data[i]["transformed_confidence"].append( response.all_confidences[i] )
                    model_data[i]["correctness"].append( gt_answer == response.all_answers[i] )
            else:
                raise ValueError("please provide a valid metric")
            
            # print(f"GT: {gt_answer}, all_ans: {response.all_answers}")

        return {"data_by_model": model_data, "final_model_idx": final_model_data}


    def calibrate(self, calibration_data: list[ModelCalibrationData]):
        for model_idx, model_calibration_data in enumerate(calibration_data):
            self.models[model_idx].conf_signal_calibrator.calibrate(
                model_calibration_data
            )
            
    def compute_calibrated_confidence(self, calibration_data: list[ModelCalibrationData]):
        """ Calibrate the transformed confidences for all models. """
        return [ 
            self.models[i].conf_signal_calibrator.calibrate_confidence_signal(
                calibration_data[i]['transformed_confidence']
            ) for i in range(len(self.models))
        ]
    
    def compute_correctness(self, calibration_data: list[ModelCalibrationData]):
        """ Compute correctness for all models. """
        return [
            np.array(calibration_data[i]['correctness']).astype(int) for i in range(len(calibration_data))
        ]

    def evaluate_calibration(
        self, calibration_data_eval: list[ModelCalibrationData], n_ece_bins=10,
        remove_inf_or_nan=False
    ):
        ece_by_model = []

        for model_idx, model in enumerate(self.models):
            conf = np.array(calibration_data_eval[model_idx]["transformed_confidence"])
            int_correctness = np.array(calibration_data_eval[model_idx]["correctness"]).astype(int)

            if remove_inf_or_nan:
                slxn_idx = (~np.isinf(conf) & ~np.isnan(conf))

            predicted_probabilities = (
                model.conf_signal_calibrator
                    .calibrate_confidence_signal(conf)
            )
            ece_output = compute_ece(
                confidences=predicted_probabilities if not remove_inf_or_nan else predicted_probabilities[slxn_idx],
                labels=int_correctness if not remove_inf_or_nan else int_correctness[slxn_idx],
                n_bins=n_ece_bins,
            )
            ece_by_model.append(ece_output["ece"])

        return ece_by_model

    def __repr__(self) -> str:
        model_strings = (
            "\n  " + "\n  ".join([str(model) for model in self.models]) + "\n"
        )
        return f"Chain(models=[{model_strings}])"


    def validate_output(
        self,
        model: Model,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_retries: Optional[int] = None,
        do_not_calibrate: bool = False,
    ) -> tuple[str, float, float]:
        """
        Attempt to get a valid output from the model within the allowed number of retries.
        """
        initial_temperature = temperature
        all_answers = []
        all_costs = []
        all_latencies = []
        all_tokens = []

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # try:
        for retry in range(max_retries):
            answer, calibrated_conf, cost, latency, num_tokens = model.answer_query(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_not_calibrate=do_not_calibrate,
            )
            all_answers.append(answer)
            all_costs.append(cost)
            all_latencies.append(latency)
            all_tokens.append(num_tokens)

            if (self.allowed_outputs is None) or answer in self.allowed_outputs:
                return answer, calibrated_conf, cost, latency, num_tokens

            if retry == max_retries - 1:
                if initial_temperature != 0:
                    temperature = 0
            elif temperature == 0:
                temperature = 1.0

        raise LLMBadOutputError(
            f"LLMBadOutputError: Attempted to get a valid output from the model within {max_retries} retries. Attempted answers: {all_answers}",
            trace=all_answers,
        )
        # except LLMBadOutputError as e:
        #     print(e)
        #     return all_answers[0], -1.0, all_costs[0], all_latencies[0], all_tokens[0]

    def answer_query(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_retries: Optional[int] = 10,
        run_all_models=False,
        override_model_parameters: list[ModelParameters] = None,
        do_not_calibrate: bool = False,
    ) -> ChainResponse:
        """
        Process a query through the chain of models and return the final response.
        """
        all_answers = []
        all_confidences = []
        all_costs = []
        all_latencies = []
        all_tokens = []
        final_answer = None
        final_model_idx = None
        final_model_name = None

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        for model_idx, model in enumerate(self.models):
            answer, calibrated_conf, cost, latency, num_tokens = self.validate_output(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_retries=max_retries,
                do_not_calibrate=do_not_calibrate,
            )

            all_answers.append(answer)
            all_confidences.append(calibrated_conf)
            all_costs.append(cost)
            all_latencies.append(latency)
            all_tokens.append(num_tokens)

            # override model parameters if necessary
            model_reject_thold = (
                model.thresholds["reject"]
                if (override_model_parameters is None)
                else override_model_parameters[model_idx]["reject"]
            )
            model_accept_thold = (
                model.thresholds["accept"]
                if (override_model_parameters is None)
                else override_model_parameters[model_idx]["accept"]
            )

            if calibrated_conf == -1.0:
                continue
            elif calibrated_conf < model_reject_thold:
                trace = ChainResponse(
                    answer=self.REJECT_QUERY_ANSWER,
                    conf=calibrated_conf,
                    cost=sum(all_costs),
                    latency=sum(all_latencies),
                    final_model_idx=model_idx,
                    final_model_name=model.model_name,
                    all_answers=all_answers,
                    all_confidences=all_confidences,
                    all_model_names=self.model_names[: model_idx + 1],
                    all_costs=all_costs,
                    all_latencies=all_latencies,
                    all_tokens=all_tokens
                )
                if not run_all_models:
                    print(trace)
                    raise LLMUncertaintyError(
                        f"Model #{model_idx} ({model.model_name}) rejected the query",
                        trace,
                    )
                else:
                    # remember current answer as final answer
                    final_answer = self.REJECT_QUERY_ANSWER
                    final_model_idx = model_idx
                    final_model_name = model.model_name

            elif calibrated_conf >= model_accept_thold:
                if not run_all_models:
                    return ChainResponse(
                        answer=answer,
                        conf=calibrated_conf,
                        cost=sum(all_costs),
                        latency=sum(all_latencies),
                        final_model_idx=model_idx,
                        final_model_name=model.model_name,
                        all_answers=all_answers,
                        all_confidences=all_confidences,
                        all_model_names=self.model_names[: model_idx + 1],
                        all_costs=all_costs,
                        all_latencies=all_latencies,
                        all_tokens=all_tokens
                    )
                else:
                    # remember current answer as final answer
                    final_answer = answer
                    final_model_idx = model_idx
                    final_model_name = model.model_name

            elif model_idx == len(self.models) - 1:
                if not run_all_models:
                    return ChainResponse(
                        answer=answer,
                        conf=calibrated_conf,
                        cost=sum(all_costs),
                        latency=sum(all_latencies),
                        final_model_idx=model_idx,
                        final_model_name=model.model_name,
                        all_answers=all_answers,
                        all_confidences=all_confidences,
                        all_model_names=self.model_names,
                        all_costs=all_costs,
                        all_latencies=all_latencies,
                        all_tokens=all_tokens
                    )
                else:
                    # if we recorded the final answer earlier, use it
                    final_answer = final_answer if final_answer is not None else answer
                    final_model_idx = (
                        final_model_idx if final_model_idx is not None else model_idx
                    )
                    final_model_name = (
                        final_model_name
                        if final_model_name is not None
                        else model.model_name
                    )

                    return ChainResponse(
                        answer=final_answer,
                        conf=calibrated_conf,
                        cost=sum(
                            [all_costs[midx] for midx in range(0, final_model_idx + 1)]
                        ),
                        latency=sum(
                            [
                                all_latencies[midx]
                                for midx in range(0, final_model_idx + 1)
                            ]
                        ),
                        final_model_idx=final_model_idx,
                        final_model_name=final_model_name,
                        all_answers=all_answers,
                        all_confidences=all_confidences,
                        all_model_names=self.model_names,
                        all_costs=all_costs,
                        all_latencies=all_latencies,
                        all_tokens=all_tokens
                    )

        ### include this case in case the final model has confidence=-1.0 (bad output case)
        if run_all_models:
            final_answer = final_answer if final_answer is not None else answer
            final_model_idx = (
                final_model_idx if final_model_idx is not None else model_idx
            )
            final_model_name = (
                final_model_name if final_model_name is not None else model.model_name
            )

            return ChainResponse(
                answer=final_answer,
                conf=calibrated_conf,
                cost=sum([all_costs[midx] for midx in range(0, final_model_idx + 1)]),
                latency=sum(
                    [all_latencies[midx] for midx in range(0, final_model_idx + 1)]
                ),
                final_model_idx=final_model_idx,
                final_model_name=final_model_name,
                all_answers=all_answers,
                all_confidences=all_confidences,
                all_model_names=self.model_names,  # Include all model names since we ran all models
                all_costs=all_costs,
                all_latencies=all_latencies,
                all_tokens=all_tokens
            )
        
    
class PreferenceChain(Chain):
    """ Chain implementation that """
    
    def __init__(
        self,
        models: List[Model],
        lambda_c: float = 1.0/1000, # cost penalty, in units of error cost
        lambda_r: float = 100.0/1000, # abstention penalty, in units of error cost
        cost_of_error = None,
        cost_of_abstention = None,
        allowed_outputs: Optional[List[str]] = None,
        rejection_symbol="[REJECT]",
        max_new_tokens=None,
    ):
        super().__init__(
            models, allowed_outputs, rejection_symbol, max_new_tokens
        )

        for model in self.models:
            model.conf_signal_calibrator = AllModelCorrectnessPredictor()

        # Allow for more intuitive setting of the penalties
        if cost_of_error is not None:
            lambda_c = 1/cost_of_error
        if cost_of_abstention is not None:
            lambda_r = cost_of_abstention/cost_of_error

        self.lambda_c = lambda_c
        self.lambda_r = lambda_r
        self.rejection_rate = None

    
    def fit_rejection_rate(
            self, 
            saved_responses: list[ChainResponse],
            lambda_c = None,
            lambda_r = None,
            rejection_rate_estimate: float = None,
            TOL = 1e-3,
            quiet=False
        ):
        """ Fit the rejection rate of the chain. """
        if rejection_rate_estimate is None:
            rej_rate_est = self.rejection_rate
        else:
            rej_rate_est = rejection_rate_estimate

        rej_rate_prev = np.inf

        # repeat until convergence
        iter_count = 0
        while np.abs(rej_rate_est - rej_rate_prev) > TOL:
            # print rejection rejection rate estimate
            if not quiet:
                print(f"[ iter {iter_count} ] rejection_rate = {rej_rate_est}")

            # assess actual rejection rate
            chain_responses = []
            for response in saved_responses:
                try:
                    chain_response = self.answer_query(
                        lambda_r=lambda_r,
                        lambda_c=lambda_c,
                        rejection_rate=rej_rate_est,
                        simulate_raw_response=response,
                        do_not_calibrate=False
                    )
                except LLMUncertaintyError as e:
                    chain_response = e.trace
                # add this response
                chain_responses.append(chain_response)

            n_rejects = len(
                [ x for x in chain_responses 
                 if x.answer == self.REJECT_QUERY_ANSWER ]
            )
            
            # save old rejection rate estimate
            rej_rate_prev = rej_rate_est
            # replace rejection rate estimate by empirical rejection rate
            rej_rate_est = n_rejects/len(saved_responses)
            # update iteration count
            iter_count += 1

        return rej_rate_est


    def calibrate(
            self,
            calibration_data: list[ModelCalibrationData],
            fit_rejection_rate=False,
            saved_responses: list[ChainResponse] = None,
            rejection_rate_estimate=0.2,
        ):
        # calibrate the predictors
        for model_idx, allmodel_calibration_data in enumerate(calibration_data):
            self.models[model_idx].conf_signal_calibrator.calibrate(
                allmodel_calibration_data
            )
        if fit_rejection_rate:
            rejection_rate = self.fit_rejection_rate(
                saved_responses, rejection_rate_estimate=rejection_rate_estimate
            )
            self.rejection_rate = rejection_rate


    def answer_query(
            self, system_prompt = None, user_prompt = None, max_new_tokens = None, 
            temperature = None, top_p = None, top_k = None, max_retries = 10, 
            run_all_models=False, do_not_calibrate = False,
            decision_mode: Literal["mean", "lwr_conf", "upr_conf"] = "mean",
            rejection_rate = None,
            lambda_c: float = None, lambda_r: float = None,
            simulate_raw_response: ChainResponse = None,
            simulate_transformed_response: ChainResponse = None,
            start_model_idx = 0,
        ):
        if lambda_c is None:
            lambda_c = self.lambda_c
        if lambda_r is None:
            lambda_r = self.lambda_r
        if rejection_rate is None:
            rejection_rate = self.rejection_rate

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Run the preference-based logic
        # Here, calibrated_conf will be a dict of predicted correctness and conf ints
        
        assert (start_model_idx >= 0) and (start_model_idx < len(self.models))
        model_idx = start_model_idx
        model_trace = []
        certain_enough = False

        all_answers = []
        all_confidences = []
        all_latencies = []
        all_costs = []
        all_tokens = []

        while not certain_enough:
            model_trace.append(model_idx)

            if simulate_transformed_response is not None:
                raise NotImplementedError("not implemented yet")

            if simulate_raw_response is None:
                answer, calibrated_conf, cost, latency, num_tokens = self.validate_output(
                    model=self.models[model_idx],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_retries=max_retries,
                    do_not_calibrate=do_not_calibrate,
                )
            else:
                answer = simulate_raw_response.all_answers[model_idx]
                raw_conf = simulate_raw_response.all_confidences[model_idx]
                transformed_conf = self.models[model_idx].conf_signal_transform.transform_confidence_signal(raw_conf)
                if do_not_calibrate:
                    calibrated_conf = transformed_conf
                else:
                    calibrated_conf = self.models[model_idx].conf_signal_calibrator.calibrate_confidence_signal(transformed_conf)
                cost = simulate_raw_response.all_costs[model_idx]
                latency = simulate_raw_response.all_latencies[model_idx]
                num_tokens = simulate_raw_response.all_tokens[model_idx]

            all_answers.append(answer)
            all_confidences.append(calibrated_conf)
            all_latencies.append(latency)
            all_costs.append(cost)
            all_tokens.append(num_tokens)

            if (calibrated_conf == -1.0) and (model_idx < len(self.models)-1): 
                model_idx += 1
                continue
            elif (calibrated_conf == -1.0) and (model_idx == len(self.models)-1):
                return ChainResponse(
                    answer=answer,
                    reject=False,
                    malformed=True,
                    model_trace=model_trace,
                    all_costs=all_costs,
                    all_answers=all_answers,
                    all_confidences=all_confidences,
                    all_latencies=all_latencies,
                    all_tokens=all_tokens
                )

            # Decide what to do:

            Model_Costs = np.array([ 
                (num_tokens['in'] * m.cpm_tokens['in'] + num_tokens['out'] * m.cpm_tokens['out'])/1e+6
                    for m in self.models
            ])
            # Set cost of this model to zero because we have already paid the price
            Model_Costs[model_idx] = 0

            assert len(calibrated_conf['confidence_intervals']) == len(self.models)
            assert len(calibrated_conf['correctness']) == len(self.models)

            if decision_mode == "mean":
                Correctness_Estimate = np.array(calibrated_conf['correctness'])
            elif decision_mode == "lwr_conf":
                Correctness_Estimate = np.array([ ci[0] for ci in calibrated_conf['confidence_intervals'] ])
            elif decision_mode == "upr_conf":
                Correctness_Estimate = np.array([ ci[1] for ci in calibrated_conf['confidence_intervals'] ])
            
            assert Model_Costs.shape == Correctness_Estimate.shape

            # print("Correctness estimate", Correctness_Estimate)
            # print("Model costs", Model_Costs)

            # Construct choices
            conditional_error = (1-Correctness_Estimate)/(1-rejection_rate)
            choices = conditional_error + lambda_c*Model_Costs
            
            lowest_penalty = np.min(choices)
            best_model_idx = np.argmin(choices)

            # Account for abstention
            if (lowest_penalty < lambda_r) and (best_model_idx != model_idx):
                if best_model_idx in model_trace:
                    # just accept the query and return
                    print(f"Prevented circular path, tried to revisit model {best_model_idx} [trace={model_trace}]")
                    return ChainResponse(
                        answer=answer,
                        reject=False,
                        malformed=False,
                        model_trace=model_trace,
                        all_costs=all_costs,
                        all_answers=all_answers,
                        all_confidences=all_confidences,
                        all_latencies=all_latencies,
                        all_tokens=all_tokens
                    )
                    # print(f"Error! Circular trace! (going to model {best_model_idx} again (trace: {model_trace}))")
                    # raise Exception("circular trace! let's not go round infinitely often")
                else:
                    model_idx = best_model_idx
            elif (lowest_penalty < lambda_r) and (best_model_idx == model_idx):
                # accept the query and return
                return ChainResponse(
                    answer=answer,
                    reject=False,
                    malformed=False,
                    model_trace=model_trace,
                    all_costs=all_costs,
                    all_answers=all_answers,
                    all_confidences=all_confidences,
                    all_latencies=all_latencies,
                    all_tokens=all_tokens
                )
            elif (lowest_penalty >= lambda_r):
                # reject the query and return
                trace = ChainResponse(
                    answer=self.REJECT_QUERY_ANSWER,
                    reject=True,
                    malformed=False,
                    model_trace=model_trace,
                    all_costs=all_costs,
                    all_answers=all_answers,
                    all_confidences=all_confidences,
                    all_latencies=all_latencies,
                    all_tokens=all_tokens
                )

                raise LLMUncertaintyError(
                        f"Model #{model_idx} ({self.models[model_idx].model_name}) rejected the query",
                        trace,
                )



if __name__ == "__main__":
    sysprompt = "Answer the question with yes or no by outputting just 'Y' or 'N'. Don't say anything else."
    userprompt = "Is the following lyrics from the song 'un poco loco': What color is the sky?\n¡Ay, mi amor! ¡Ay, mi amor!\nYou tell me that it's red\n¡Ay, mi amor! ¡Ay, mi amor!"

    fireworks_chain = Chain(
        models=[
            Model(
                model_name="llama3.2-1b",
                thresholds={"reject": 0.2, "accept": 0.99},
                client=FireworksClient(),
            ),
            Model(
                model_name="llama3.2-3b",
                thresholds={"reject": 0.2, "accept": 0.99},
                client=FireworksClient(),
            ),
            Model(
                model_name="llama3-8b",
                thresholds={"reject": 0.2, "accept": 0.99},
                client=FireworksClient(),
            ),
            Model(
                model_name="llama3-70b",
                thresholds={"reject": 0.2, "accept": 0.99},
                client=FireworksClient(),
            ),
            Model(
                model_name="llama3.1-405b",
                client=FireworksClient(),
            ),
        ],
        allowed_outputs=["Y", "N"],
    )

    openai_chain = Chain(
        models=[
            Model(
                model_name="gpt-4o-mini",
                thresholds={"reject": 0.2, "accept": 0.99},
                client=OpenAIClient(),
            ),
            Model(
                model_name="gpt-4o",
                client=OpenAIClient(),
            ),
        ],
        allowed_outputs=["Y", "N"],
    )

    chain_response = openai_chain.answer_query(
        sysprompt, userprompt, temperature=1.0, max_retries=10
    )
    print("Answer:", chain_response.answer)
    print("Confidence:", chain_response.confidence)
    print("Total Cost:", chain_response.cost)
    print("Final Model Index:", chain_response.final_model_idx)
    print("Final Model Name:", chain_response.final_model_name)
    print("All Answers:", chain_response.all_answers)
    print("All Confidences:", chain_response.all_confidences)
    print("All Model Names:", chain_response.all_model_names)
    print("All Costs:", chain_response.all_costs)
