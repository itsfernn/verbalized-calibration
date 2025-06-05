from utils.utils import (
    extract_answer,
    extract_texts_and_confidences,
)


class MessageTree:
    def __init__(self, response, role, parent=None):
        self.response = response
        self.role = role
        try:
            self.text, self.confidence = extract_texts_and_confidences(response)[0]
        except ValueError:
            self.text = response
            self.confidence = 1.0

        self.answer = extract_answer(response) is not None
        self.parent = parent
        self.children = []

    def to_dict(self):
        return {
            "response": self.response,
            "role": self.role,
            "children": [child.to_dict() for child in self.children],
        }

    def __str__(self):
        return str(self.to_dict())

    def depth(self):
        return self.parent.depth() + 1 if self.parent else 0

    def add_child(self, message, role):
        new_child = MessageTree(message, role, parent=self)
        self.children.append(new_child)
        return new_child

    def get_message_history(self):
        messages = []
        current_node = self
        while current_node:
            messages.append(
                {"content": current_node.response, "role": current_node.role}
            )
            current_node = current_node.parent
        return messages[::-1]

    def get_aggregated_confidence(self):
        aggregated_confidence = 1
        current_node = self
        while current_node:
            aggregated_confidence *= current_node.confidence
            current_node = current_node.parent
        return aggregated_confidence
