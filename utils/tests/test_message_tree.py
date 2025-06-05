import pytest
from utils.message_tree import MessageTree


def test_initialization():
    response = "Hello, how can I help you?"
    role = "user"
    message_tree = MessageTree(response, role)

    assert message_tree.response == response
    assert message_tree.role == role
    assert message_tree.text == "Hello, how can I help you?"
    assert message_tree.confidence == 1.0
    assert message_tree.parent is None
    assert message_tree.children == []


def test_add_child():
    parent_response = "Hello, how can I help you?"
    child_response = "I need assistance with my order."
    parent_tree = MessageTree(parent_response, "user")
    child_tree = parent_tree.add_child(child_response, "assistant")

    assert len(parent_tree.children) == 1
    assert parent_tree.children[0] == child_tree
    assert child_tree.parent == parent_tree


def test_get_message_history():
    root_response = "Hello, how can I help you?"
    child_response = "I need assistance with my order."
    message_tree = MessageTree(root_response, "user")
    child_node = message_tree.add_child(child_response, "assistant")

    history = child_node.get_message_history()
    assert len(history) == 2
    assert history[0] == {"content": root_response, "role": "user"}
    assert history[1] == {"content": child_response, "role": "assistant"}


def test_get_aggregated_confidence():
    root_response = "Hello, how can I help you?"
    child_response = "I need assistance with my order."
    message_tree = MessageTree(root_response, "user")
    child_tree = message_tree.add_child(child_response, "assistant")

    # Test with default confidence of 1.0
    assert message_tree.get_aggregated_confidence() == 1.0

    # Modify the confidence of the child
    child_tree.confidence = 0.9
    assert child_tree.get_aggregated_confidence() == 0.9

    # Add another child with a different confidence
    another_child = child_tree.add_child("Another message.", "assistant")
    another_child.confidence = 0.8
    assert pytest.approx(another_child.get_aggregated_confidence()) == pytest.approx(
        0.72
    )
