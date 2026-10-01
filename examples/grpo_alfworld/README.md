# GRPO on ALFWorld Dataset

This example shows the usage of GRPO on the ALFWorld dataset.

For more detailed information, please refer to the [documentation](https://github.com/agentscope-ai/Trinity-RFT/blob/65139711219da4338c954e546c4b8434e4f75ec2/docs/sphinx_doc/source/tutorial/example_multi_turn.md).

The config file is located in [`alfworld.yaml`](alfworld.yaml).

NOTE: For the Concatenated Multi-Turn RFT setup in the Qwen-2.5 series, the model may not follow the `<think></think><action></action>` format. You may need to perform SFT first, then GRPO.

The SFT data should be named as `<TRINITY_SFT_DATASET_PATH>/data.json`, following the format:

```
[
    {
         "messages": [
            {
                "role": "system", # fixed, align with the grpo workflow: alfworld_workflow.
                "content": "\nYou are an agent interacting with a virtual text-based environment.\n\n## Notes:\nAt each step, you should first think then perform action to fulfill the instruction. You should ALWAYS wrap your thinking with the   tag and wrap your action with the   tag.\nYou should ALWAYS take one action each step. \nYou should finish the task and buy the item within 15 steps.\nDO NOT try to interact with the user at anytime. Finish the task and buy the item by yourself.\n\n## Action Format:\nBelow are the available commands you can use:\n  look:                             look around your current location\n  inventory:                        check your current inventory(you can only have 1 item in your inventory)\n  go to (receptacle):               move to a receptacle\n  open (receptacle):                open a receptacle\n  close (receptacle):               close a receptacle\n  take (object) from (receptacle):  take an object from a receptacle\n  move (object) to (receptacle):  place an object in or on a receptacle\n  examine (something):              examine a receptacle or an object\n  use (object):                     use an object\n  heat (object) with (receptacle):  heat an object using a receptacle\n  clean (object) with (receptacle): clean an object using a receptacle\n  cool (object) with (receptacle):  cool an object using a receptacle\n  slice (object) with (object):     slice an object using a sharp object\n\nFor example your output should be like this:\n To solve the task, I need first to ... go to cabinet 1\n"
            },
            {
                "role": "user",
                "content": "Observation: {observation by alfworld}"
            },
            {
                "role": "assistant",
                "content": "<think>think process</think><action>action</action>"
            },
            {
                "role": "user",
                "content": "Observation: {observation by alfworld}"
            },
            {
                "role": "assistant",
                "content": "<think>think process</think><action>action</action>"
            },
            .....
        ],
    },
    {
        "messages": [
            {
                ......
            },
        ]
    },
    {
         "messages": [
            {
                .......
            },
        ]
    },
]
```
