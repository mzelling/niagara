from openai import OpenAI
from .api_client import APIClient, ModelNotFoundError
from time import time


class OpenAIClient(APIClient):
    def __init__(self):
        super().__init__(api_key_env_name="OPENAI_API_KEY", model_type="openai")
        self.client = OpenAI(api_key=self.api_key)

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
                logprobs=True,
                **{
                    k: v
                    for k, v in {
                        "temperature": temperature,
                        "max_tokens": max_new_tokens,
                        "top_p": top_p
                    }.items()
                    if v is not None
                },
            )
            stop_time = time()
            network_latency = 1000*(stop_time - start_time) # measured in milliseconds
        except Exception as e:
            print(f"OpenAI API call failed: {str(e)}")
            raise e

        usage_info = response.usage
        input_tokens = usage_info.prompt_tokens
        output_tokens = usage_info.completion_tokens
        total_cost = self.calculate_cost(model_info, input_tokens, output_tokens)
        num_tokens = {"in": input_tokens, "out": output_tokens}

        content = response.choices[0].message.content
        token_logprobs = [
            token.logprob for token in response.choices[0].logprobs.content
        ] if (model_info["path"] not in {"o1-preview", "o1-mini"}) else []
        return content, token_logprobs, total_cost, network_latency, num_tokens


if __name__ == "__main__":
    openai_client = OpenAIClient()
    response = openai_client.get_answer(
        "gpt-4o-mini", "You are a helpful assistant.", "What is the capital of France?"
    )
    print("Response:", OpenAIClient.get_response_text(response))
    print("Token logprobs:", OpenAIClient.get_token_logprobs(response))
    print("Max logprob:", OpenAIClient.get_max_logprob(response))
    print("Cost:", OpenAIClient.get_cost(response))
