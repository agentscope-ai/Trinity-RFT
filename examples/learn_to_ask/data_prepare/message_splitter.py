import json


def split_single_message_list(messages):
    """
    Split a single message list into multiple rounds.

    Args:
        messages (list): List of message dictionaries with 'role' and 'content' keys

    Returns:
        list: List of rounds, where each round contains messages and remaining chat

    Note: consecutive same-role messages are preserved as-is (RealMedConv is IM-style).
    """
    rounds = []
    round_number = 1
    i = 0

    while i < len(messages):
        # Advance i past one [non-user]* [user]* block (one "turn")
        while i < len(messages) and messages[i].get("role") != "user":
            i += 1
        while i < len(messages) and messages[i].get("role") == "user":
            i += 1

        # Use full prefix as round messages (paper Sec 3.1: C_{t-1} = (u_0, ..., u_{t-1}))
        round_messages = messages[:i]
        remaining_messages = messages[i:]
        round_entry = {"round_number": round_number, "messages": round_messages}

        # Add remaining chat if there are remaining messages
        if remaining_messages:
            remaining_chat_parts = []
            for msg in remaining_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                remaining_chat_parts.append(f"{role}: {content}")
            round_entry["remaining_chat"] = "\n".join(remaining_chat_parts)
        else:
            round_entry["remaining_chat"] = ""

        rounds.append(round_entry)
        round_number += 1

    return rounds


def split_session_to_json_lines(session):
    """
    Split a session dictionary into multiple rounds and convert to JSON lines.

    Args:
        session (dict): Session dictionary containing 'session_id', 'diagn', and 'messages' keys
            - session_id (str): Session identifier
            - diagn (str): Diagnosis information
            - messages (list): List of message dictionaries with 'role' and 'content' keys

    Returns:
        list: List of JSON strings, each representing a round with cid, session_id, diagn, messages, and remaining_chat
    """
    rounds = split_single_message_list(session["messages"])

    json_lines = []
    for round_data in rounds:
        round_entry = {
            "cid": f"{session['session_id']}_{round_data['round_number']}",
            "session_id": session["session_id"],
            "diagn": session["diagn"],
            "messages": round_data["messages"],
            "remaining_chat": round_data["remaining_chat"],
        }

        json_lines.append(json.dumps(round_entry, ensure_ascii=False))

    return json_lines


# Example usage:
if __name__ == "__main__":
    # Example of splitting a single message list
    example_messages = [
        {"role": "assistant", "content": "Hello, how can I help you today?"},
        {"role": "user", "content": "I've been having headaches lately."},
        {"role": "assistant", "content": "How long have you been experiencing these headaches?"},
        {"role": "user", "content": "For about a week now."},
        {"role": "assistant", "content": "I see. Have you taken any medication for them?"},
        {"role": "user", "content": "Yes, I've tried some over-the-counter pain relievers."},
    ]

    example_session = {"session_id": "session_1", "diagn": "migraine", "messages": example_messages}
    json_lines = split_session_to_json_lines(example_session)
    print("JSON lines output:")
    for i, line in enumerate(json_lines):
        print(f"Line {i + 1}: {line}")
