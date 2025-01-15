import numpy as np
from abc import ABC
from typing import Optional, List, Tuple
from .exceptions import LLMBadOutputError


class ConfidenceSignal(ABC):
    """Abstract base class for generating confidence signals."""

    def get_confidence_signal(
        self,
        model: Optional[any] = None,
        answer: Optional[str] = None,
        logprobs: Optional[List[float]] = None,
        prev_conf: Optional[List[float]] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Calculate the confidence signal based on model outputs."""
        pass

    def __repr__(self):
        return self.__class__.__name__
    

class ModelIntrinsicLogProb(ConfidenceSignal):
    """ Pass the mean of the logprobs through. """
    
    def get_confidence_signal(
        self,
        model=None,
        answer=None,
        logprobs: list[float] = None,
        prev_conf=None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> float:
        """ Simply pass the mean of the logprobs through. """
        if logprobs is None or len(logprobs) < 1:
            raise ValueError("Logprobs list must contain at least one element.")
        if max_new_tokens == 1:
            assert len(logprobs) == 1, f"There should be only 1 logprob, but we found {len(logprobs)}"
        return (np.mean(np.array(logprobs)), 0, 0)
    

class ModelIntrinsicProb(ConfidenceSignal):
    """Confidence signal based on the intrinsic probabilities of model's answers."""

    def get_confidence_signal(
        self,
        model=None,
        answer=None,
        logprobs: list[float] = None,
        prev_conf=None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> float:
        """Calculate the confidence signal from log probabilities."""
        if logprobs is None or len(logprobs) < 1:
            raise ValueError("Logprobs list must contain at least one element.")
        return (np.exp(np.mean(np.array(logprobs))), 0.0, 0.0)
    

class ConsistencyBasedConfidence(ConfidenceSignal):
    """
    Repeatedly sample frmo the model and assess consistency between the different
    responses.
    """
    
    def __init__(self, k=10):
        pass

    def get_confidence_signal(
            self, model = None, answer = None, 
            logprobs = None, prev_conf = None, 
            system_prompt = None, user_prompt = None,
            max_new_tokens = None, temperature = None, 
            top_p = None, top_k = None
        ):
        
        # Step 1: generate k answers

        # Step 2: check which answers are equivalent

        # Step 3: compute entropy on the coarse-grained distribution
        pass


class AskModelConfidence(ConfidenceSignal):
    """
    Obtain confidence estimate by re-prompting model to ask if the proposed answer
    is correct.
    """

    ASK_MODEL_SYSPROMPT = """Consider the instructions and query below, then evaluate whether the proposed response is a correct response to the query. Output 'Y' if the response is completely correct, truthful, and accurate, otherwise output 'N'. Don't say anything else."""
    ASK_MODEL_USERPROMPT_TEMPLATE = """Instructions: {task_system_prompt}\n\nQuery: {task_user_prompt}\n\nProposed Response: {proposed_response}\n\nIs the proposed response completely correct, truthful, and accurate? """

    def __init__(
        self,
        eval_system_prompt=ASK_MODEL_SYSPROMPT,
        eval_user_prompt_template=ASK_MODEL_USERPROMPT_TEMPLATE,
        extract_fun=None,
        temperature=0.0,
        top_p=1.0,
        top_k=50,
        max_retries=10,
        allowed_outputs={"Y", "N", "y", "n"},
        max_new_tokens=1
    ):
        self.eval_system_prompt = eval_system_prompt
        self.eval_user_prompt_template = eval_user_prompt_template
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_retries = max_retries
        self.extract_fun = extract_fun
        self.allowed_outputs = allowed_outputs
        self.max_new_tokens = max_new_tokens


    @staticmethod
    def validate_raw_output(
        model,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_retries: Optional[int] = 10,
        allowed_outputs = None
    ) -> tuple[str, float, float]:
        """
        Attempt to get a valid output from the model within the allowed number of retries.
        """
        initial_temperature = temperature
        all_answers = []
        all_costs = []
        all_latencies = []
        all_tokens = []

        try:
            for retry in range(max_retries):
                answer, logprobs, cost, latency, num_tokens = model.query_raw_model(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                all_answers.append(answer)
                all_costs.append(cost)
                all_latencies.append(latency)
                all_tokens.append(num_tokens)

                if (allowed_outputs is None) or answer in allowed_outputs:
                    return answer, logprobs, cost, latency, num_tokens

                if retry == max_retries - 1:
                    if initial_temperature != 0:
                        temperature = 0
                elif temperature == 0:
                    temperature = 1.0

            raise LLMBadOutputError(
                f"LLMBadOutputError: Attempted to get a valid output from the model within {max_retries} retries. Attempted answers: {all_answers}",
                trace=all_answers,
            )
        except LLMBadOutputError as e:
            print(e)
            return all_answers[0], [-np.inf], all_costs[0], all_latencies[0], all_tokens[0]


    def get_confidence_signal(
        self,
        model=None,
        answer=None,
        logprobs=None,
        prev_conf=None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):  
        if self.max_new_tokens is not None:
            max_new_tokens = self.max_new_tokens

        eval_answer, eval_logprobs, eval_cost, eval_latency, num_tokens = self.validate_raw_output(
            model=model,
            system_prompt=self.eval_system_prompt,
            user_prompt=self.eval_user_prompt_template.format(
                task_system_prompt=system_prompt,
                task_user_prompt=user_prompt,
                proposed_response=answer,
            ),
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_retries=self.max_retries,
            allowed_outputs=self.allowed_outputs
        )

        # Extract the actual final verification answer
        if self.extract_fun is not None:
            extracted_answer = self.extract_fun(eval_answer, answer)
            if extracted_answer not in {"Y", "N", "y", "n"}:
                extracted_answer = "N"
                eval_logprobs=[-np.inf]
                print(f"model {model.model_name} did not yield extractable confidence verification!")
                print("BEGIN UNEXTRACTABLE")
                print(eval_answer)
                print("END UNEXTRACTABLE")
        else:
            extracted_answer = eval_answer

        if isinstance(eval_logprobs, float):
            eval_logprobs = [eval_logprobs]

        # logprobs should contain answer logprob and stop token logprob
        if extracted_answer[0].lower() == "y":
            return (eval_logprobs[0], eval_cost, eval_latency)
        elif extracted_answer[0].lower() == "n":
            return (np.log(1 - np.exp(eval_logprobs[0])), eval_cost, eval_latency)
        else:
            raise LLMBadOutputError("evaluation result should be one of 'Y' or 'N'.", (eval_answer, eval_logprobs, eval_cost, eval_latency))
