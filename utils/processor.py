import math
from sentence_transformers import CrossEncoder
import statistics
from utils.chat import Chat
from utils.message_tree import MessageTree
from utils.utils import (
    extract_texts_and_confidences,
    compute_f1,
    compute_exact_match,
)


# Base Processor Class
class BaseSampleProcessor:
    prompt = None  # Static class variable to hold the prompt
    few_shot_messages = []

    def __init__(self, llm_config={}):
        self.llm_config = llm_config

    def process_sample(self, sample):
        question = sample["question"]
        chat = Chat(self.prompt, **self.llm_config)
        for fs_question, fs_response in self.few_shot_messages:
            chat.add_message("The question is: " + fs_question, "user")
            chat.add_message(fs_response, "assistant")
        chat.add_message("The question is: " + question, "user")
        response = chat.response()
        texts, confidences = extract_texts_and_confidences(response)

        # retry if parsing failed
        for _ in range(2):
            if len(texts) == 0:
                response = chat.response()
                texts, confidences = extract_texts_and_confidences(response)
            else:
                break

        sample["response"] = response
        sample["texts"] = texts
        sample["confidences"] = confidences

        return sample

    def eval_sample(self, sample):
        prediction = sample["texts"][-1]
        confidence = sample["confidences"][-1]
        sample["prediction"] = str(prediction)
        sample["confidence"] = float(confidence)
        sample["f1"] = float(compute_f1(prediction, sample["answer"]))
        sample["em"] = int(compute_exact_match(prediction, sample["answer"]))
        return sample

    # Implementing the __call__ function for the Base class to call process_sample
    def __call__(self, sample):
        return self.process_sample(sample)


# Concrete Processor for Direct Verb
class DirectProcessor(BaseSampleProcessor):
    prompt = (
        "Provide your best guess for the following multi-hop question. "
        "Give your anwer with no other words or explanation. "
        "Also provide how confident you are that the answer is correct. (between 0.0 and 1.0)\n\n"
        "<format>\n"
        "Answer: {most likely guess, as short as possible; not a complete sentence, just the guess!} "
        "Confidence: {confidence for the answer between 0.0 and 1.0} \n"
        "</format>\n"
    )


class TopKProcessor(BaseSampleProcessor):
    def __init__(self, llm_config, k=5, normalize=False):
        super().__init__(llm_config)
        self.k = k
        self.normalize = normalize

        self.prompt = (
            f"Provide your top {self.k} best guesses for the following multi-hop question. "
            "Give your anwers with no other words or explanation. "
            "Also provide how confident you are that each answer is correct. (between 0.0 and 1.0)\n\n"
            "<format>\n"
            "Answer: {first guess, as short as possible; not a complete sentence, just the guess!} "
            "Confidence: {confidence for the first answer between 0.0 and 1.0} \n"
            "Answer: {second guess, as short as possible; not a complete sentence, just the guess!} "
            "Confidence: {confidence for the second answer between 0.0 and 1.0} \n"
            "...\n"
            "</format>\n"
        )

    def eval_sample(self, sample):
        matches = zip(sample["texts"], sample["confidences"])

        matches = sorted(matches, key=lambda x: x[1], reverse=True)

        prediction = matches[0][0]
        confidence = matches[0][1]
        sample["prediction"] = str(prediction)
        sample["confidence"] = float(confidence)
        sample["f1"] = float(compute_f1(prediction, sample["answer"]))
        sample["em"] = int(compute_exact_match(prediction, sample["answer"]))

        if self.normalize:
            sample["confidence"] = confidence / sum(sample["confidences"])

        return sample


# Concrete Processor for Cot Verb
class CotProcessor(BaseSampleProcessor):
    prompt = (
        "For the following multi-hop question let's think step by step. "
        "After your reasoning give your final answer on a new line with no other words or explanation. "
        "Also provide how confident you are the answer is correct. (0.0 to 1.0)\n\n"
        "<format>\n"
        "{thoughts}\n"
        "Final Answer: {most likely guess, as short as possible; not a complete sentence, just the guess!} "
        "Confidence: {confidence for your answer}\n"
        "</format>\n"
    )


# Concrete Processor for Cot Verb
class MultistepProcessor(BaseSampleProcessor):
    def __init__(self, llm_config, k=5):
        super().__init__(llm_config)
        self.k = k
        self.prompt = (
            "Provide your best guess for the following multi-hop question. "
            f"Provide a step-by-step explanation of your thought process. "
            "Each step should only contain a single new fact and be on a new line. "
            "When enough information is present, give your final answer on a new line with no other words or explanation. "
            "Also provide how confident you are between 0.0 and 1.0 that a given line is correct.\n\n"
            "<format>\n"
            "Step 1: {first reasoning step} Confidence: {confidence for step 1}\n"
            "Step 2: {second reasoning step} Confidence: {confidence for step 2}\n"
            "...\n"
            "Final Answer: {most likely guess, as short as possible; not a complete sentence, just the guess!} "
            "Confidence: {final confidence for your answer}\n"
            "</format>\n"
        )

    def eval_sample(self, sample):
        sample = super().eval_sample(sample)
        sample["confidence"] = math.prod(sample["confidences"])
        return sample


# Concrete Processor for Cot Verb
class MultistepFewshotProcessor(MultistepProcessor):
    def __init__(self, llm_config, k=5):
        super().__init__(llm_config)
        self.few_shot_messages = [
            (
                "In what county is the university at which Walter Oechel teaches located?",
                """Step 1: Walter Oechel is a biologist and a professor known for his work in the field of ecology and climate change. Confidence: 0.8 
Step 2: He is associated with San Diego State University. Confidence: 0.9  
Step 3: San Diego State University is located in San Diego, California. Confidence: 0.95  
Step 4: San Diego is within San Diego County. Confidence: 0.95  
Final Answer: San Diego County Confidence: 0.95""",
            ),
            (
                "Chairmen of the Bored is an album that references a joke made by a cast member of what series?",
                """Step 1: The phrase "Chairmen of the Bored" is a play on words related to the term "Chairman of the Board." Confidence: 0.36
Step 2: Chairmen of the Bored" is an album title associated with a musical group, most likely a parody group. Confidence: 0.32
Step 3: The musical group is likely "The Lonely Island," known for digital shorts and parody songs. Confidence: 0.34
Step 4: The digital shorts created by "The Lonely Island" are frequently featured on "Saturday Night Live" (SNL). Confidence: 0.38
Step 5: A cast member of "Saturday Night Live" famously utilized a similar joke or phrase on the show, influences jokes throughout. Confidence: 0.36
Final Answer: Saturday Night Live Confidence: 0.38""",
            ),
        ]


# Concrete Processor for Cot Verb
class MultistepBeamSearchProcessor(MultistepProcessor):
    def __init__(self, k, llm_config, branching_factor, beam_width, search_criterion):
        """
        Args:
            llm_config (dict): LLM parameters.
            branching_factor (int): The number of branches to explore at each step.
            beam_width (int): The number of top candidates to keep at each level.
            search_criterion (str): The criterion used to evaluate nodes:
                ['highest_confidence',
                'median_confidence',
                'highest_aggregated_confidence',
                'median_aggregated_confidence',
                'entailment']
        """
        super().__init__(llm_config, k)
        ## assign beam search parameters
        self.branching_factor = branching_factor
        self.beam_width = beam_width
        self.search_criterion = search_criterion
        if self.search_criterion == "entailment":
            self.nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

    def entailment_score(self, premise: str, sentences: list[str]):
        assert self.nli_model is not None, "NLI model not initialized"
        sentence_pairs = [
            (premise, conclusion) for conclusion in sentences if premise != conclusion
        ]
        scores = self.nli_model.predict(sentence_pairs)  # type: ignore
        return scores[:, 1].mean()

    ## assign beam search parameters
    def select_nodes(self, nodes: list[MessageTree], k: int):
        if self.search_criterion == "highest_confidence":
            nodes.sort(key=lambda x: x.confidence, reverse=True)
        elif self.search_criterion == "median_confidence":
            pass
            ## median_confidence = statistics.median([node.confidence for node in nodes])
            # nodes.sort(key=lambda x: abs(x.confidence - median_confidence))
        elif self.search_criterion == "highest_aggregated_confidence":
            nodes.sort(key=lambda x: x.get_aggregated_confidence(), reverse=True)
        elif self.search_criterion == "median_aggregated_confidence":
            pass
            # median_confidence = statistics.median(
            ##[node.get_aggregated_confidence() for node in nodes]
            # )
        # nodes.sort(
        #     key=lambda x: abs(x.get_aggregated_confidence() - median_confidence)
        # )
        elif self.search_criterion == "entailment":
            sentences = [node.text for node in nodes]
            nodes.sort(
                key=lambda x: self.entailment_score(x.text, sentences), reverse=True
            )
        else:
            raise ValueError(f"Invalid search criterion: {self.search_criterion}")

        return nodes[:k]

    def process_sample(self, sample):
        question, answer = sample["question"], sample["answer"]
        root = MessageTree(self.prompt, "system")
        head = root.add_child("The question is: " + question, "user")
        chat = Chat(**self.llm_config, stop="\n")

        next_nodes = [head]
        prediction_nodes = []
        for _ in range(self.k):
            for node in next_nodes:
                chat.chat = node.get_message_history()
                for _ in range(self.branching_factor):
                    response = chat.response()
                    node.add_child(response, "assistant")
            children = [child for node in next_nodes for child in node.children]
            selected_nodes = self.select_nodes(
                children, self.beam_width - len(prediction_nodes)
            )
            prediction_nodes += [node for node in selected_nodes if node.answer]
            next_nodes = [node for node in selected_nodes if not node.answer]

            if len(prediction_nodes) >= self.beam_width:
                break

        # Force an awnswer for the selected_nodes if not enought answers have been generated
        if len(prediction_nodes) < self.beam_width:
            for node in next_nodes:
                children = [child for node in next_nodes for child in node.children]
                selected_nodes = self.select_nodes(
                    children, self.beam_width - len(prediction_nodes)
                )
                prediction_nodes += [node for node in selected_nodes if node.answer]
                next_nodes = [node for node in selected_nodes if not node.answer]

                if len(prediction_nodes) >= self.beam_width:
                    break

        if len(prediction_nodes) == 0:
            return {"tree": head.to_dict()}
            # raise ValueError("Unable to parse response")

        prediction_node = self.select_nodes(prediction_nodes, 1)[0]

        response = "\n".join(
            [message["content"] for message in prediction_node.get_message_history()]
        )
        prediction = prediction_node.text
        confidence = prediction_node.get_aggregated_confidence()

        return {
            "response": response,
            "prediction": prediction,
            "confidence": confidence,
            "tree": head.to_dict(),
            "f1": compute_f1(prediction, answer),
            "em": compute_exact_match(prediction, answer),
        }


# Factory Method for creating processors
def get_processor(method_name, llm_config) -> BaseSampleProcessor:
    processors = {
        "direct": lambda: DirectProcessor(llm_config),
        "multistep": lambda: MultistepProcessor(llm_config),
        "multistep-fewshot": lambda: MultistepProcessor(llm_config),
        "cot": lambda: CotProcessor(llm_config),
        "top-k": lambda: TopKProcessor(llm_config),
        "top-k-norm": lambda: TopKProcessor(llm_config, normalize=True),
    }

    processor = processors.get(method_name)

    if processor is None:
        raise ValueError(f"Invalid method: {method_name}")

    return processor()
