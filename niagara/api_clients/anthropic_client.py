from anthropic import Anthropic
from .api_client import APIClient, ModelNotFoundError
from time import time


class AnthropicClient(APIClient):
    def __init__(self):
        super().__init__(api_key_env_name="ANTHROPIC_API_KEY", model_type="anthropic")
        self.client = Anthropic(api_key=self.api_key)

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
            response = self.client.messages.create(
                model=model_info["path"],
                max_tokens=1024 if max_new_tokens is None else max_new_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                **{
                    k: v
                    for k, v in {
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                    }.items()
                    if v is not None
                },
            )
            stop_time = time()
            network_latency = 1000*(stop_time - start_time) # measured in milliseconds
        except Exception as e:
            print(f"Anthropic API call failed: {str(e)}")
            raise e

        usage_info = response.usage
        input_tokens = usage_info.input_tokens
        output_tokens = usage_info.output_tokens
        total_cost = self.calculate_cost(model_info, input_tokens, output_tokens)
        num_tokens = {"in": input_tokens, "out": output_tokens}

        content = response.content[0].text
        token_logprobs = []
        return content, token_logprobs, total_cost, network_latency, num_tokens


if __name__ == "__main__":
    anthropic_client = AnthropicClient()
    response = anthropic_client.get_answer(
        "claude-3-sonnet",
        "You are a helpful assistant.",
        "What is the capital of France?",
    )
    print("Response:", AnthropicClient.get_response_text(response))
    print("Token logprobs:", AnthropicClient.get_token_logprobs(response))
    print("Cost:", AnthropicClient.get_cost(response))
