"""
We use this script to create the huggingface format dataset files for the alfworld dataset.
NOTE: You need to install the alfworld dataset in first: https://github.com/alfworld/alfworld
"""
import argparse
import glob
import json
import os
import random

random.seed(42)


def create_dataset_files(game_data_path, output_dir, train_size=None, test_size=None):
    # get all matched game files from train and valid_seen directories
    train_game_files = glob.glob(
        os.path.expanduser(f"{game_data_path}/json_2.1.1/train/*/*/game.tw-pddl")
    )
    test_game_files = glob.glob(
        os.path.expanduser(f"{game_data_path}/json_2.1.1/valid_seen/*/*/game.tw-pddl")
    )

    # Fail before writing any output if the root is wrong or the download is
    # incomplete. Empty JSONL files otherwise hide this setup error until Ray.
    if not train_game_files or not test_game_files:
        raise ValueError(
            "No ALFWorld game files found for train or valid_seen. "
            "Set ALFWORLD_DATA/--game_data_path to the dataset root containing "
            f"json_2.1.1/ and logic/, not json_2.1.1 itself: {game_data_path}"
        )

    # get absolute path
    train_game_files = [os.path.abspath(file) for file in train_game_files]
    test_game_files = [os.path.abspath(file) for file in test_game_files]
    train_game_files = sorted(train_game_files)
    test_game_files = sorted(test_game_files)

    print(f"Total train game files found: {len(train_game_files)}")
    print(f"Total test game files found: {len(test_game_files)}")

    # if size is None, use all files
    if train_size is None:
        train_size = len(train_game_files)
    if test_size is None:
        test_size = len(test_game_files)

    # check sizes
    assert train_size <= len(
        train_game_files
    ), f"train_size {train_size} > available {len(train_game_files)}"
    assert test_size <= len(
        test_game_files
    ), f"test_size {test_size} > available {len(test_game_files)}"

    # randomly select the game files
    selected_train_files = random.sample(train_game_files, train_size)
    selected_test_files = random.sample(test_game_files, test_size)

    # make the output directory
    os.makedirs(output_dir, exist_ok=True)

    # create train and test data
    train_data = [
        {"game_file": game_file_path, "target": ""} for game_file_path in selected_train_files
    ]
    test_data = [
        {"game_file": game_file_path, "target": ""} for game_file_path in selected_test_files
    ]

    # create dataset_dict
    dataset_dict = {"train": train_data, "test": test_data}

    for split, data in dataset_dict.items():
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        with open(output_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    # create dataset_dict.json
    dataset_info = {
        "citation": "",
        "description": "Custom dataset",
        "splits": {
            "train": {"name": "train", "num_examples": len(train_data)},
            "test": {"name": "test", "num_examples": len(test_data)},
        },
    }

    with open(os.path.join(output_dir, "dataset_dict.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Created dataset with {len(train_data)} train and {len(test_data)} test examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_data_path", type=str, default=None, required=False)
    parser.add_argument("--local_dir", type=str, default=None, required=False)
    parser.add_argument("--train_size", type=int, default=None, required=False)
    parser.add_argument("--test_size", type=int, default=None, required=False)
    args = parser.parse_args()

    if args.game_data_path is None:
        # ALFWORLD_DATA is the dataset path in the environment variable
        # you need to set it when install alfworld dataset
        from alfworld.info import ALFWORLD_DATA

        args.game_data_path = ALFWORLD_DATA

    if args.local_dir is None:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        args.local_dir = f"{current_file_dir}/alfworld_data"

    # use all data by default, or specify train_size and test_size if needed
    create_dataset_files(
        game_data_path=args.game_data_path,
        output_dir=args.local_dir,
        train_size=args.train_size,
        test_size=args.test_size,
    )
