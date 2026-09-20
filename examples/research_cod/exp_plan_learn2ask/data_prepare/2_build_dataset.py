import argparse
import json
import os


def process_message(json_obj):
    info_set = json_obj.get("info_set")
    info_set_str = ", ".join(info_set) if isinstance(info_set, list) else ""
    if "user: " not in json_obj["remaining_chat"]:
        decision_str = "stop"
    else:
        decision_str = "continue"
    # KNOWN BIAS: drops 2607 continue tasks (extractor ignores medication / allergy / chronic).
    if not info_set_str and decision_str == "continue":
        if_keep = False
    else:
        if_keep = True
    return if_keep, info_set_str, decision_str


def main(input_file_path, output_file_path):
    # Drop user-ending sessions (no doctor reply -> no ground-truth final decision)
    # and ghost rounds (remaining_chat empty -> decision point already passed).
    by_session = {}
    with open(input_file_path, "r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line.strip())
            by_session.setdefault(data["session_id"], []).append(data)

    n_sess_dropped = n_ghost_dropped = n_info_dropped = n_kept = 0
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for sid, rounds in by_session.items():
            rounds.sort(key=lambda r: r.get("round_number", 0))
            last_msgs = rounds[-1].get("messages") or []
            if last_msgs and last_msgs[-1].get("role") == "user":
                n_sess_dropped += 1
                continue
            for data in rounds:
                if not data.get("remaining_chat"):
                    n_ghost_dropped += 1
                    continue
                if_keep, info_set, decision = process_message(data)
                if not if_keep:
                    n_info_dropped += 1
                    continue

                new_item = {
                    "cid": data["cid"],
                    "session_id": data["session_id"],
                    "diagn": data["diagn"],
                    "messages": data["messages"],
                    "decision_truth": decision,
                    "info_truth": info_set,
                }
                outfile.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                n_kept += 1
    print(
        f"job done! kept={n_kept}, dropped: session={n_sess_dropped}, "
        f"ghost_round={n_ghost_dropped}, no_info={n_info_dropped}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_template",
        type=str,
        default="examples/research_cod/data/learn2ask_artifacts/{split}_processed.jsonl",
    )
    parser.add_argument(
        "--output_template",
        type=str,
        default="examples/research_cod/data/learn2ask/{split}.jsonl",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"])

    args = parser.parse_args()

    for split in args.splits:
        input_path = args.input_template.replace("{split}", split)
        output_path = args.output_template.replace("{split}", split)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"[{split}] {input_path} -> {output_path}")
        main(input_path, output_path)
