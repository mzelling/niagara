import json
import os
from typing import Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv

load_dotenv()


class APIClient:
    ONE_MILLION = 1000000

    def __init__(self, api_key_env_name: str, model_type: str):
        """Initialize the API client with necessary credentials or settings."""
        self.api_key = os.getenv(api_key_env_name)
        if not self.api_key:
            raise ValueError(
                f"Missing API key: {api_key_env_name} environment variable not set"
            )
        self.models = self.load_models(model_type)

    @staticmethod
    def load_models(model_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Load model configurations from JSON file.
        """
        models_path = os.path.join(os.path.dirname(__file__), "../models.json")
        try:
            with open(models_path) as f:
                all_models = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading models configuration: {str(e)}")
            return {}

        model_info = all_models.get(model_type, {})
        metadata_models = {}

        for model_name, details in model_info.items():
            try:
                if not isinstance(details, list) or len(details) != 3:
                    print(f"Warning: Invalid model configuration for {model_name}")
                    continue

                metadata_models[model_name] = {
                    "path": str(details[0]),
                    "input_cost": float(details[1]),
                    "output_cost": float(details[2]),
                }
            except (IndexError, ValueError) as e:
                print(f"Error processing model {model_name}: {str(e)}")
                continue
        return metadata_models

    @staticmethod
    def get_response_text(response_tuple):
        return response_tuple[0]

    @staticmethod
    def get_token_logprobs(response_tuple):
        return response_tuple[1]

    @staticmethod
    def get_cost(response_tuple):
        return response_tuple[2]

    @staticmethod
    def get_max_logprob(response_tuple):
        token_logprobs = response_tuple[1]
        return max(token_logprobs) if token_logprobs is not None else None

    @staticmethod
    def calculate_input_cost(model_info, num_tokens):
        input_cost_per_token = model_info.get("input_cost", 0)
        return num_tokens * input_cost_per_token / APIClient.ONE_MILLION

    @staticmethod
    def calculate_output_cost(model_info, num_tokens):
        output_cost_per_token = model_info.get("output_cost", 0)
        return num_tokens * output_cost_per_token / APIClient.ONE_MILLION

    @staticmethod
    def calculate_cost(
        model_info: Dict[str, Any],
        num_input_tokens: Union[int, float],
        num_output_tokens: Union[int, float],
    ) -> float:
        input_cost = APIClient.calculate_input_cost(model_info, num_input_tokens)
        output_cost = APIClient.calculate_output_cost(model_info, num_output_tokens)
        return input_cost + output_cost
    
    def __repr__(self):
        return self.__class__.__name__

    def get_answer(
        self,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[str, list[float], float]:
        """
        Get an answer from the API. Must be implemented by subclasses.
        
        Returns:
            Tuple containing (response_text, token_logprobs, cost, network_latency, num_tokens)
        """
        raise NotImplementedError("Subclasses must implement get_answer method")

class ModelNotFoundError(Exception):
    def __init__(self, model_name):
        message = f"Model name '{model_name}' not found in the models list."
        super().__init__(message)
