from chat import Chat
import math
from utils.utils import (
    extract_answer_and_confidence,
    compute_f1,
    compute_exact_match,
)


# Base Processor Class
class BaseSampleProcessor:
    prompt = None  # Static class variable to hold the prompt

    def __init__(self, llm_config={}):
        self.llm_config = llm_config

    def process_sample(self, sample):
        raise NotImplementedError("Subclasses should implement this method.")

    # Implementing the __call__ function for the Base class to call process_sample
    def __call__(self, sample):
        return self.process_sample(sample)


# Concrete Processor for Direct Verb
class DirectProcessor(BaseSampleProcessor):
    prompt = (
        "Provide your best guess for the following multi-hop question. "
        "Give your anwer with no other words or explanation. "
        "Also provide how confident you are that the answer is correct. (between 0.0 and 1.0)\n\n"
        "Format:\n\n"
        "Answer: <most likely guess, as short as possible; not a complete sentence, just the guess!> "
        "Confidence: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>"
    )

    def process_sample(self, sample):
        question, answer = sample["question"], sample["answer"]
        chat = Chat(self.prompt, **self.llm_config)
        chat.add_message("The question is: " + question, "user")
        response = chat.response()
        prediction, confidence = extract_answer_and_confidence(response)

        return {
            "response": response,
            "prediction": prediction,
            "confidence": confidence,
            "f1": compute_f1(prediction, answer),
            "em": compute_exact_match(prediction, answer),
        }


# Concrete Processor for Cot Verb
class CotProcessor(BaseSampleProcessor):
    prompt = (
        "Provide your best guess for the following multi-hop question. "
        "Let's think step by step. "
        "After your reasoning give your final answer on a new line with no other words or explanation. "
        "Also provide how confident you are the the answer is correct. (0.0 to 1.0)\n\n"
        "Format:\n\n"
        "The question is: <Multi Hop Question>\n"
        "<Thoughts>\n"
        "Final Answer: <most likely guess, as short as possible; not a complete sentence, just the guess!> Confidence: <confidence score for overall question between 0 and 1>"
    )

    def process_sample(self, sample):
        question, answer = sample["question"], sample["answer"]
        chat = Chat(self.prompt, **self.llm_config)
        chat.add_message("The question is: " + question, "user")
        response = chat.response()
        prediction, confidence = extract_answer_and_confidence(response)

        return {
            "response": response,
            "prediction": prediction,
            "confidence": confidence,
            "f1": compute_f1(prediction, answer),
            "em": compute_exact_match(prediction, answer),
        }


# Concrete Processor for Cot Verb
class MultistepProcessor(BaseSampleProcessor):
    prompt = (
        "Provide your best guess for the following multi-hop question. "
        "Provide a step-by-step explanation of your thought process. (0-5 reasoning steps) "
        "Each step should only contain a single new fact and be on a new line. "
        "When enough information is present, give your final answer on a new line with no other words or explanation. "
        "Also provide how confident you are between 0 and 1 that a given line is correct.\n\n"
        "Format:\n"
        "The question is: <Multi Hop Question>\n"
        "<step 1 of thought process> Confidence: <confidence score for step 1>\n"
        "<step 2 of thought process> Confidence: <confidence score for step 2>\n"
        "...\n"
        "Final Answer: <most likely guess, as short as possible; not a complete sentence, just the guess!> Confidence: <confidence score for overall question between 0 and 1>"
    )

    def process_sample(self, sample):
        question, answer = sample["question"], sample["answer"]
        chat = Chat(self.prompt, **self.llm_config)
        chat.add_message("The question is: " + question, "user")
        response = chat.response()
        reasoning_steps = response.split("\n")

        confidences = []
        prediction = None
        for step in reasoning_steps:
            answer, confidence = extract_answer_and_confidence(step)
            if confidence:
                confidences.append(confidence)
            if answer:
                prediction = answer

        confidence = math.prod(confidences)

        return {
            "response": response,
            "prediction": prediction,
            "confidence": confidence,
            "f1": compute_f1(prediction, answer),
            "em": compute_exact_match(prediction, answer),
        }


# Factory Method for creating processors
def get_processor(method_name):
    processors = {
        "cot": CotProcessor,
        "direct": DirectProcessor,
        "multistep": MultistepProcessor,
    }

    processor = processors.get(method_name)

    if processor is None:
        raise ValueError(f"Invalid method: {method_name}")

    return processor
