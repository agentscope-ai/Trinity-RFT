import unittest
from copy import deepcopy
from typing import List

import torch

from trinity.buffer.operators.experience_operator import create_operators
from trinity.common.config import OperatorConfig
from trinity.common.experience import EID, Experience


def get_experiences(task_num: int, repeat_times: int = 1, step_num: int = 1) -> List[Experience]:
    """Generate a list of experiences for testing."""
    return [
        Experience(
            eid=EID(task=i, run=j, step=k),
            tokens=torch.zeros((5,)),
            prompt_length=4,
            reward=j,
            logprobs=torch.tensor([0.1]),
            info={
                "llm_quality_score": i,
                "llm_difficulty_score": k,
            },
        )
        for i in range(task_num)
        for j in range(repeat_times)
        for k in range(step_num)
    ]


class TestRewardShapingMapper(unittest.IsolatedAsyncioTestCase):
    async def test_basic_usage(self):
        # test input cache
        op_configs = [
            OperatorConfig(
                name="reward_shaping_mapper",
                args={
                    "reward_shaping_configs": [
                        {
                            "stats_key": "llm_quality_score",
                            "op_type": "ADD",
                            "weight": 1.0,
                        },
                        {
                            "stats_key": "llm_difficulty_score",
                            "op_type": "MUL",
                            "weight": 0.5,
                        },
                    ]
                },
            )
        ]
        ops = create_operators(op_configs)
        self.assertEqual(len(ops), 1)

        op = ops[0]
        task_num = 8
        repeat_times = 4
        step_num = 2
        experiences = get_experiences(
            task_num=task_num, repeat_times=repeat_times, step_num=step_num
        )
        res_exps, metrics = await op.process(deepcopy(experiences))
        self.assertEqual(len(res_exps), task_num * repeat_times * step_num)
        self.assertIn("reward_diff/mean", metrics)
        self.assertIn("reward_diff/min", metrics)
        self.assertIn("reward_diff/max", metrics)

        for prev_exp, res_exp in zip(experiences, res_exps):
            self.assertAlmostEqual(
                (prev_exp.reward + prev_exp.info["llm_quality_score"])
                * 0.5
                * prev_exp.info["llm_difficulty_score"],
                res_exp.reward,
            )
