# -*- coding: utf-8 -*-
"""Test cases for Storage modules."""
import os
import pickle
import unittest

import torch

from trinity.buffer.schema.sql_schema import ExperienceModel
from trinity.common.experience import EID, CustomField, Experience

db_url = os.path.join(os.path.dirname(__file__), "tmp", "test.db")
dataset_path = os.path.join(os.path.dirname(__file__), "data")


class TestEID(unittest.TestCase):
    def test_eid_properties(self):
        # test properties
        eid = EID(batch=1, task=2, run=3, step=4, suffix="abc123")
        self.assertEqual(eid.uid, "1/2/3/4/abc123")
        self.assertEqual(eid.sid, "1/2/4")
        self.assertEqual(eid.rid, "1/2/3")
        self.assertEqual(eid.tid, "1/2")
        self.assertEqual(str(eid), "1/2/3/4/abc123")
        self.assertIn("EID(batch=1, task=2, run=3, step=4, uuid=abc123)", repr(eid))

        # test unique
        eid1 = EID(batch=1, task=2, run=3, step=4)
        eid2 = EID(batch=1, task=2, run=3, step=4)
        self.assertNotEqual(eid1.suffix, eid2.suffix)
        self.assertNotEqual(eid1.uid, eid2.uid)

        # test default
        eid = EID()
        eid2 = EID()
        self.assertIsInstance(eid.suffix, str)
        self.assertEqual(eid.batch, "")
        self.assertEqual(eid.task, "")
        self.assertEqual(eid.run, 0)
        self.assertEqual(eid.step, 0)
        self.assertNotEqual(eid.uid, eid2.uid)


class TestExperience(unittest.TestCase):
    def test_single_turn_experience(self):
        tokens = torch.tensor([10, 11, 12], dtype=torch.int32)
        logprobs = torch.tensor([0.2, 0.3], dtype=torch.float32)
        exp = Experience(tokens=tokens, logprobs=logprobs, reward=1.0, prompt_length=1)
        self.assertEqual(exp.experience_type, "single_turn")
        self.assertTrue(torch.equal(exp.tokens, tokens))
        self.assertTrue(torch.equal(exp.logprobs, logprobs))
        self.assertEqual(exp.reward, 1.0)
        self.assertEqual(exp.prompt_length, 1)
        self.assertTrue(torch.equal(exp.action_mask, torch.tensor([1, 1], dtype=torch.bool)))

    def test_multi_turn_experience(self):
        tokens = torch.tensor([1, 2, 3, 4])
        logprobs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        action_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        exp = Experience(tokens=tokens, logprobs=logprobs, reward=2.0, action_mask=action_mask)
        self.assertEqual(exp.experience_type, "multi_turn")
        self.assertTrue(torch.equal(exp.action_mask, action_mask))
        self.assertEqual(exp.prompt_length, 1)

    def test_dpo_experience(self):
        tokens = torch.tensor([1, 2])
        chosen = torch.tensor([3, 4])
        rejected = torch.tensor([5, 6])
        exp = Experience(tokens=tokens, chosen=chosen, rejected=rejected, reward=0.5)
        self.assertEqual(exp.experience_type, "dpo")
        self.assertTrue(torch.equal(exp.chosen, chosen))
        self.assertTrue(torch.equal(exp.rejected, rejected))
        self.assertEqual(exp.prompt_length, 2)

    def test_serialize_deserialize(self):
        tokens = torch.tensor([1, 2, 3])
        exp = Experience(tokens=tokens, reward=1.23, prompt_length=1)
        data = exp.serialize()
        exp2 = Experience.deserialize(data)
        self.assertTrue(torch.equal(exp.tokens, exp2.tokens))
        self.assertEqual(exp.reward, exp2.reward)
        self.assertEqual(exp.prompt_length, exp2.prompt_length)

    def test_serialize_many_deserialize_many(self):
        exp1 = Experience(
            eid=EID(batch=1, task=1, run=1, step=1),
            tokens=torch.tensor([1, 2, 3], dtype=torch.int32),
            logprobs=torch.tensor([0.1, 0.2], dtype=torch.float32),
            reward=1.0,
            prompt_length=1,
            info={"source": "a"},
            metrics={"m": 1.0},
            multi_modal_inputs={"image": torch.randn(2, 3)},
            custom_fields=[
                CustomField(
                    source_field="foo",
                    destination_field="bar",
                    data_type=torch.float32,
                )
            ],
        )
        exp2 = Experience(
            eid=EID(batch=1, task=1, run=2, step=1),
            tokens=torch.tensor([4, 5, 6, 7], dtype=torch.int32),
            reward=2.0,
            prompt_length=2,
            info={"source": "b"},
            metrics={"m": 2.0},
        )

        data = Experience.serialize_many([exp1, exp2])
        restored = Experience.deserialize_many(data)

        self.assertEqual(len(restored), 2)
        self.assertTrue(torch.equal(restored[0].tokens, exp1.tokens))
        self.assertTrue(torch.equal(restored[0].logprobs, exp1.logprobs))
        self.assertEqual(restored[0].reward, exp1.reward)
        self.assertEqual(restored[0].info, exp1.info)
        self.assertEqual(restored[0].metrics, exp1.metrics)
        self.assertIsNotNone(restored[0].multi_modal_inputs)
        self.assertIn("image", restored[0].multi_modal_inputs)
        self.assertTrue(
            torch.equal(restored[0].multi_modal_inputs["image"], exp1.multi_modal_inputs["image"])
        )
        self.assertEqual(len(restored[0].custom_fields), 1)
        self.assertEqual(restored[0].custom_fields[0].destination_field, "bar")

        self.assertTrue(torch.equal(restored[1].tokens, exp2.tokens))
        self.assertEqual(restored[1].reward, exp2.reward)
        self.assertEqual(restored[1].prompt_length, exp2.prompt_length)

    def test_serialize_many_with_shared_multimodal_tensor(self):
        shared_pixel_values = torch.randn(2, 3)
        shared_image_grid_thw = torch.tensor([1, 2, 3], dtype=torch.int64)
        exps = []

        for i in range(4):
            exps.append(
                Experience(
                    eid=EID(batch=1, task=1, run=i, step=1),
                    tokens=torch.tensor([1, 2, 3], dtype=torch.int32),
                    prompt_length=1,
                    multi_modal_inputs={
                        "pixel_values": shared_pixel_values,
                        "image_grid_thw": shared_image_grid_thw,
                    },
                )
            )

        data = Experience.serialize_many(exps)
        restored = Experience.deserialize_many(data)

        self.assertEqual(len(restored), 4)
        for exp in restored:
            self.assertIsNotNone(exp.multi_modal_inputs)
            self.assertTrue(
                torch.equal(exp.multi_modal_inputs["pixel_values"], shared_pixel_values)
            )
            self.assertTrue(
                torch.equal(exp.multi_modal_inputs["image_grid_thw"], shared_image_grid_thw)
            )

    def test_deserialize_legacy_pickle_payload(self):
        exp = Experience(tokens=torch.tensor([1, 2, 3]), reward=1.23, prompt_length=1)
        legacy_data = pickle.dumps(exp)
        restored = Experience.deserialize(legacy_data)
        self.assertTrue(torch.equal(restored.tokens, exp.tokens))
        self.assertEqual(restored.reward, exp.reward)

    def test_deserialize_single_rejects_batch_payload(self):
        exp1 = Experience(tokens=torch.tensor([1, 2, 3]), prompt_length=1)
        exp2 = Experience(tokens=torch.tensor([4, 5, 6]), prompt_length=1)
        payload = Experience.serialize_many([exp1, exp2])
        with self.assertRaises(ValueError):
            Experience.deserialize(payload)

    def test_to_dict(self):
        tokens = torch.tensor([1, 2, 3])
        exp = Experience(
            tokens=tokens, reward=2.5, prompt_length=1, prompt_text="hi", response_text="yo"
        )
        d = exp.to_dict()
        self.assertIn("eid", d)
        self.assertIn("type", d)
        self.assertIn("reward", d)
        self.assertEqual(d["prompt_text"], "hi")
        self.assertEqual(d["response_text"], "yo")
        self.assertEqual(d["reward"], 2.5)

    def test_action_mask_and_logprobs_type(self):
        exp = Experience(tokens=[1, 2, 3], logprobs=[0.1, 0.2, 0.3], prompt_length=1)
        self.assertIsInstance(exp.tokens, torch.Tensor)
        self.assertIsInstance(exp.logprobs, torch.Tensor)
        self.assertIsInstance(exp.action_mask, torch.Tensor)

    def test_assertions(self):
        # prompt_length must be > 0
        with self.assertRaises(AssertionError):
            Experience(tokens=[1, 2, 3], prompt_length=0)
        # tokens must be larger than prompt_length for single-turn
        with self.assertRaises(AssertionError):
            Experience(tokens=[1, 2], prompt_length=2)
        # DPO: tokens must match prompt_length
        exp = Experience(tokens=[1, 2], chosen=[3], rejected=[4], prompt_length=1)
        exp.prompt_length = 2  # should automatically adjust

    def test_hf_datasets_conversion(self):
        import torch

        from trinity.common.experience import (
            Experience,
            from_hf_datasets,
            to_hf_datasets,
        )

        exps = [
            Experience(
                eid=EID(batch=1, task=2, run=3, step=4),
                tokens=torch.tensor([1, 2, 3]),
                reward=1.0,
                logprobs=torch.tensor([0.2, 0.3]),
                prompt_length=1,
                advantages=None,
                returns=None,
                info={"key": "value"},
                metrics={"accuracy": 0.9},
                action_mask=torch.tensor([1, 1]),
            ),
            Experience(
                eid=EID(batch=1, task=5, run=6, step=7),
                tokens=torch.tensor([4, 5, 6]),
                reward=2.0,
                logprobs=torch.tensor([0.5]),
                prompt_length=2,
                advantages=torch.tensor([0.9]),
                returns=torch.tensor([0.9]),
                info={"key": "value"},
                metrics={"accuracy": 0.95},
                action_mask=torch.tensor([1]),
            ),
        ]

        ds = to_hf_datasets(exps)
        self.assertEqual(len(ds), 2)

        exps2 = from_hf_datasets(ds)
        self.assertEqual(len(exps2), 2)

        for e1, e2 in zip(exps, exps2):
            self.assertTrue(torch.equal(e1.tokens, e2.tokens))
            self.assertEqual(e1.reward, e2.reward)
            self.assertTrue(torch.equal(e1.logprobs, e2.logprobs))
            self.assertEqual(e1.prompt_length, e2.prompt_length)
            self.assertTrue(torch.equal(e1.action_mask, e2.action_mask))
            self.assertEqual(e1.eid.uid, e2.eid.uid)
            self.assertEqual(e1.info, e2.info)
            self.assertEqual(e1.metrics, e2.metrics)
            if e1.advantages is not None:
                self.assertTrue(torch.equal(e1.advantages, e2.advantages))
            if e1.returns is not None:
                self.assertTrue(torch.equal(e1.returns, e2.returns))


class TestExperienceConversion(unittest.TestCase):
    """Test cases for ExperienceModel"""

    def test_experience_model_experience_conversion(self):
        """Test the conversion between Experience and ExperienceModel"""
        tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
        reward = 0.6
        prompt_length = 2
        logprobs = torch.tensor([0, 0, 0.1], dtype=torch.float32)
        experience = Experience(
            tokens=tokens,
            reward=reward,
            prompt_length=prompt_length,
            logprobs=logprobs,
            info={"model_version": 0},
        )

        model = ExperienceModel.from_experience(experience)
        new_experience = model.to_experience()
        self.assertTrue(torch.equal(new_experience.tokens, tokens))
        self.assertEqual(new_experience.prompt_length, prompt_length)
        self.assertEqual(new_experience.reward, reward)
        self.assertTrue(torch.equal(new_experience.logprobs, logprobs))
        self.assertTrue(torch.equal(new_experience.action_mask, experience.action_mask))


if __name__ == "__main__":
    unittest.main()
