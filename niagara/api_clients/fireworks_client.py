import numpy as np
from fireworks.client import Fireworks
from .api_client import APIClient, ModelNotFoundError
from time import time


class FireworksClient(APIClient):
    def __init__(self):
        super().__init__(api_key_env_name="FIREWORKS_API_KEY", model_type="fireworks")
        self.client = Fireworks(api_key=self.api_key)

    def get_answer(
        self,
        model_name=None,
        system_prompt=None,
        user_prompt=None,
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
    ):
        model_info = self.models.get(model_name)
        if model_info is None:
            raise ModelNotFoundError(model_name)

        try:
            start_time = time()
            response = self.client.chat.completions.create(
                model=model_info["path"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                logprobs=1,
                **{
                    k: v
                    for k, v in {
                        "temperature": temperature,
                        "max_tokens": max_new_tokens,
                        "top_p": top_p,
                        "top_k": top_k,
                    }.items()
                    if v is not None
                },
            )
            stop_time = time()
            network_latency = 1000*(stop_time - start_time) # measured in milliseconds
        except Exception as e:
            print(f"Fireworks API call failed: {str(e)}")
            raise e

        usage_info = response.usage
        input_tokens = usage_info.prompt_tokens
        output_tokens = usage_info.completion_tokens
        total_cost = self.calculate_cost(model_info, input_tokens, output_tokens)
        num_tokens = {"in": input_tokens, "out": output_tokens}

        content = response.choices[0].message.content

        # tokens = response.choices[0].logprobs.tokens
        # token_ids = response.choices[0].logprobs.token_ids

        token_logprobs = response.choices[0].logprobs.token_logprobs
        clipped_token_logprobs = np.clip(token_logprobs, -np.inf, 0.0)
        return content, clipped_token_logprobs, total_cost, network_latency, num_tokens


if __name__ == "__main__":
    fireworks_client = FireworksClient()
    response = fireworks_client.get_answer(
        "llama3-8b",
        "You are a helpful assistant.",
        "What is the capital of France?",
    )
    # response = fireworks_client.get_answer(
    #     "llama-v3-8b-instruct",
    #     "You are a helpful assistant.",
    #     "You have chosen to abstain from this query because you are too uncertain, explain why you are uncertain: Is the following lyrics from the song 'un poco loco': What color is the sky?\n¡Ay, mi amor! ¡Ay, mi amor!\nYou tell me that it's red\n¡Ay, mi amor! ¡Ay, mi amor!",
    # )
    print("Response:", FireworksClient.get_response_text(response))
    print("Token logprobs:", FireworksClient.get_token_logprobs(response))
    print("Max logprob:", FireworksClient.get_max_logprob(response))
    print("Cost:", FireworksClient.get_cost(response))
