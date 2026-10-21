import unittest

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding

from trinity.common.config import AlgorithmConfig
from trinity.trainer.verl.losses import TrinityPolicyLoss


class TestTrinityPolicyLossMicroBatching(unittest.TestCase):
    """veRL's engine sums the micro-batch losses, so the gradient must not depend on the split."""

    batch_size, prompt_len, max_response_len = 8, 3, 5

    def setUp(self):
        torch.manual_seed(0)
        B, P, R = self.batch_size, self.prompt_len, self.max_response_len
        self.response_lens = torch.tensor([5, 2, 4, 1, 3, 5, 2, 4])
        self.response_mask = (torch.arange(R)[None] < self.response_lens[:, None]).long()
        self.prompts = torch.randint(1, 50, (B, P))
        self.responses = torch.randint(1, 50, (B, R))
        self.attention_mask = torch.cat([torch.ones(B, P, dtype=torch.long), self.response_mask], 1)
        self.advantages = torch.linspace(-1.0, 1.0, B)[:, None].expand(B, R).clone()
        self.logits = torch.randn(B, R, requires_grad=True)

    def _micro_batch(self, idx):
        data = TensorDict(
            {
                "prompts": self.prompts[idx],
                "responses": self.responses[idx],
                "input_ids": torch.cat([self.prompts[idx], self.responses[idx]], 1),
                "attention_mask": self.attention_mask[idx],
                "position_ids": (self.attention_mask[idx].cumsum(-1) - 1).clamp(min=0),
                "response_mask": self.response_mask[idx],
                "old_log_probs": torch.full((len(idx), self.max_response_len), -1.0),
                "advantages": self.advantages[idx],
            },
            batch_size=[len(idx)],
        )
        data = left_right_2_no_padding(data)
        # what veRL's engine attaches to every micro-batch of a mini-batch
        tu.assign_non_tensor(
            data,
            batch_num_tokens=int(self.response_mask.sum()),
            dp_size=1,
            global_batch_size=self.batch_size,
        )
        # unpadded per-token log-probs (prompt + response, shifted by one)
        log_probs = torch.nested.as_nested_tensor(
            [
                torch.cat(
                    [
                        torch.zeros(self.prompt_len - 1),
                        -torch.nn.functional.softplus(self.logits[i, : self.response_lens[i]]),
                        torch.zeros(1),
                    ]
                )
                for i in idx
            ],
            layout=torch.jagged,
        )
        return {"log_probs": log_probs}, data

    def _gradient(self, loss_fn, num_micro_batches):
        self.logits.grad = None
        total = 0
        for start in range(num_micro_batches):
            idx = list(range(start, self.batch_size, num_micro_batches))
            loss, _ = loss_fn(*self._micro_batch(idx))
            total = total + loss
        total.backward()
        return self.logits.grad.clone()

    def test_gradient_independent_of_micro_batches(self):
        for loss_agg_mode in ("token-mean", "seq-mean-token-mean"):
            config = AlgorithmConfig(
                policy_loss_fn="ppo",
                policy_loss_fn_args={"loss_agg_mode": loss_agg_mode, "clip_range": 0.2},
                kl_loss_fn="none",
                kl_loss_fn_args={},
                entropy_loss_fn="none",
                entropy_loss_fn_args={"entropy_coef": 0.0},
                loss_agg_mode=loss_agg_mode,
            )
            loss_fn = TrinityPolicyLoss(config)
            reference = self._gradient(loss_fn, 1)
            for num_micro_batches in (2, 4):
                with self.subTest(loss_agg_mode=loss_agg_mode, num_micro_batches=num_micro_batches):
                    torch.testing.assert_close(
                        self._gradient(loss_fn, num_micro_batches), reference
                    )
