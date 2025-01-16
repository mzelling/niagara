# Niagara
    
        __    __    __    __
       /  \__/  \__/  \__/  \
       |  |  |  |  |  |  |  |
       |  |  |  |  |  |  |  |
       |░░|  |░░|  |░░|  |░░|
       |░░|  |░░|  |░░|  |░░|
       |░░|  |░░|  |░░|  |░░|
       |░░|  |░░|  |░░|  |░░|
       |░░|  |░░|  |░░|  |░░|
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ~~~   ~~~   ~~~   ~~~   ~~~
    
Niagara is a Python package for composing LLM cascades using arbitrary LLMs available via public APIs.

### Example

```python

import numpy as np
from niagara import Chain, Model
from niagara.confidence_signals import AskModelConfidence
from niagara.api_clients import OpenAIClient, FireworksClient

my_chain = Chain(models=[
    Model(
        model_name="llama3.1-8b", 
        conf_signal=AskModelConfidence(),
        thresholds={"accept": np.log(0.9), "reject": np.log(0.0)}, 
        client=FireworksClient()
    ),
    Model(
        model_name="qwen2.5-72b-instruct",
        conf_signal=AskModelConfidence(),
        thresholds={"accept": np.log(0.8), "reject": np.log(0.0)},
        client=FireworksClient()
    ),
    Model(
        model_name="gpt-4o", 
        conf_signal=AskModelConfidence(),
        thresholds={"accept": np.log(0.0), "reject": np.log(0.0)}, 
        client=OpenAIClient())
])

response = my_chain.answer_query(
    system_prompt = "Pretend you are indignant and refuse to answer the question, posing a math problem instead.",
    user_prompt = "What is the capital of France?",
    max_new_tokens = 200,
    temperature = 0.0,
)

print(f"Final model: {my_chain.model_names[response.final_model_idx]}")
print(f"Final output: {response.all_answers[response.final_model_idx]}")

```

Output:
```
Final model: gpt-4o
Final output: I refuse to answer such a simple question! Instead, solve this: What is the sum of the first 100 positive integers?
```
