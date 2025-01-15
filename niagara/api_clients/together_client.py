import numpy as np
from together import Together
from .api_client import APIClient, ModelNotFoundError
from time import time


class TogetherClient(APIClient):
    def __init__(self):
        super().__init__(api_key_env_name="TOGETHER_API_KEY", model_type="together")
        self.client = Together(api_key=self.api_key)

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
            print(f"Together API call failed: {str(e)}")
            raise e

        usage_info = response.usage
        input_tokens = usage_info.prompt_tokens
        output_tokens = usage_info.completion_tokens
        total_cost = self.calculate_cost(model_info, input_tokens, output_tokens)
        num_tokens = {"in": input_tokens, "out": output_tokens}

        content = response.choices[0].message.content
        token_logprobs = response.choices[0].logprobs.token_logprobs
        clipped_token_logprobs = np.clip(token_logprobs, -np.inf, 0.0)
        return content, clipped_token_logprobs, total_cost, network_latency, num_tokens


if __name__ == "__main__":
    together_client = TogetherClient()
    answer, logprobs, cost, latency = together_client.get_answer(
        model_name="llama3.1-8b-turbo",
        system_prompt="Consider the instructions and query below, then evaluate whether the proposed response is a correct response to the query. Output 'Y' if the response is completely correct, truthful, and accurate, otherwise output 'N'. Don't say anything else.",
        user_prompt="Instructions: You are a helpful assistant.\n\nQuery: Answer the trivia question below with just the answer. Be as concise as possible. Don't answer with a sentence, explain your answer, or say anything else.\n\nQuestion: Which Star Wars character said the famous line \"Hello There\"?\n\nAnswer:\n\nProposed Response: General Kenobi\n\nIs the proposed response completely correct, truthful, and accurate?",
        max_new_tokens=1,
        temperature=0.0,
        top_p=1.0,
        top_k=50,
    )
    # print("Response:", TogetherClient.get_response_text(response))
    # print("Token logprobs:", TogetherClient.get_token_logprobs(response))
    # print("Max logprob:", TogetherClient.get_max_logprob(response))
    # print("Cost:", TogetherClient.get_cost(response))

    if answer[0].lower() == "y":
        print(np.exp(logprobs[0]))
    elif answer[0].lower() == "n":
        print(1 - np.exp(logprobs[0]))
    else:
        print("evaluation result should be one of 'Y' or 'N'.", (answer, logprobs, cost, latency))
