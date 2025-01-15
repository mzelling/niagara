class LLMUncertaintyError(Exception):
    """ 
    Throw this exception when a model in the chain 
    rejects the query (abstains). 
    """
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace

class LLMBadOutputError(Exception):
    """
    Throw this exception when the model's output does
    not match the list of allowed outputs (as in multiple-choice QA).
    """
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace