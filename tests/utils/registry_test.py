# -*- coding: utf-8 -*-
"""Test cases for workflows registry mapping."""
import unittest
from unittest.mock import MagicMock

import ray
import torch

from trinity.algorithm import (
    ADVANTAGE_FN,
    ALGORITHM_TYPE,
    ENTROPY_LOSS_FN,
    KL_FN,
    POLICY_LOSS_FN,
    SAMPLE_STRATEGY,
    AdvantageFn,
    AlgorithmType,
    EntropyLossFn,
    KLFn,
    PolicyLossFn,
    SampleStrategy,
)
from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.operators import EXPERIENCE_OPERATORS, ExperienceOperator
from trinity.buffer.reader import READER
from trinity.buffer.schema import FORMATTER, SQL_SCHEMA
from trinity.buffer.selector import SELECTORS, BaseSelector
from trinity.buffer.storage import PRIORITY_FUNC
from trinity.buffer.storage.queue import PriorityFunction
from trinity.common.rewards import REWARD_FUNCTIONS, RewardFn
from trinity.common.workflows import WORKFLOWS, Task, Workflow
from trinity.utils.monitor import MONITOR, Monitor


@ENTROPY_LOSS_FN.register_module("dummy_entropy_loss_fn")
class DummyEntropyLossFn(EntropyLossFn):
    def __init__(self, entropy_coef: float):
        self.entropy_coef = entropy_coef

    def __call__(
        self,
        entropy,
        action_mask,
        **kwargs,
    ):
        return torch.tensor(0.0), {}


class ImportUtils:
    def run(self):
        from trinity.common.workflows import WORKFLOWS, Workflow

        workflow_cls = WORKFLOWS.get("tests.utils.plugins.main.MainDummyWorkflow")
        assert issubclass(workflow_cls, Workflow)
        workflow = workflow_cls(task=Task(), model=MagicMock())
        res = workflow.run()
        assert res[0] == 0
        assert res[1] == "0"


class TestRegistryWithRay(unittest.TestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)

    def tearDown(self):
        ray.shutdown()

    def test_dynamic_import(self):
        # test local import
        ImportUtils().run()
        # test remote import
        ray.get(ray.remote(ImportUtils).remote().run.remote())


class TestRegistry(unittest.TestCase):
    """Test registry functionality."""

    def test_common_module_registry_mapping(self):
        """Test registry mapping in common module"""
        # test workflow
        workflow_names = list(WORKFLOWS._default_mapping.keys())
        for workflow_name in workflow_names:
            with self.subTest(workflow_name=workflow_name):
                workflow_cls = WORKFLOWS.get(workflow_name)
                self.assertIsNotNone(
                    workflow_cls, f"{workflow_name} should be retrievable from registry"
                )
                self.assertTrue(
                    issubclass(workflow_cls, Workflow),
                    f"{workflow_name} should be a subclass of Workflow",
                )
        with self.assertRaises(ValueError):
            WORKFLOWS.get("non_existent_workflow")

        # test reward function
        reward_fn_names = list(REWARD_FUNCTIONS._default_mapping.keys())
        for reward_fn_name in reward_fn_names:
            with self.subTest(reward_fn_name=reward_fn_name):
                reward_fn_cls = REWARD_FUNCTIONS.get(reward_fn_name)
                self.assertIsNotNone(
                    reward_fn_cls, f"{reward_fn_name} should be retrievable from registry"
                )
                self.assertTrue(
                    issubclass(reward_fn_cls, RewardFn),
                    f"{reward_fn_name} should be a subclass of RewardFn",
                )
        with self.assertRaises(ValueError):
            REWARD_FUNCTIONS.get("non_existent_reward_fn")

    def test_algorithm_registry_mapping(self):
        """Test registry mapping in algorithm module"""
        # test algorithm
        algorithm_names = list(ALGORITHM_TYPE._default_mapping.keys())
        for algorithm_name in algorithm_names:
            with self.subTest(algorithm_name=algorithm_name):
                algorithm_cls = ALGORITHM_TYPE.get(algorithm_name)
                self.assertIsNotNone(
                    algorithm_cls, f"{algorithm_name} should be retrievable from registry"
                )
                self.assertTrue(
                    issubclass(algorithm_cls, AlgorithmType),
                    f"{algorithm_name} should be a subclass of AlgorithmType",
                )
        with self.assertRaises(ValueError):
            ALGORITHM_TYPE.get("non_existent_algorithm")

        # test advantage function
        advantage_fn_names = list(ADVANTAGE_FN._default_mapping.keys())
        for advantage_fn_name in advantage_fn_names:
            with self.subTest(advantage_fn_name=advantage_fn_name):
                advantage_fn_cls = ADVANTAGE_FN.get(advantage_fn_name)
                self.assertIsNotNone(
                    advantage_fn_cls, f"{advantage_fn_name} should be retrievable from registry"
                )
                self.assertTrue(
                    issubclass(advantage_fn_cls, AdvantageFn),
                    f"{advantage_fn_name} should be a subclass of AdvantageFn",
                )
        with self.assertRaises(ValueError):
            ADVANTAGE_FN.get("non_existent_advantage_fn")

        # test entropy loss function
        entropy_loss_fn_names = list(ENTROPY_LOSS_FN._default_mapping.keys())
        for entropy_loss_fn_name in entropy_loss_fn_names:
            with self.subTest(entropy_loss_fn_name=entropy_loss_fn_name):
                entropy_loss_fn_cls = ENTROPY_LOSS_FN.get(entropy_loss_fn_name)
                self.assertIsNotNone(
                    entropy_loss_fn_cls,
                    f"{entropy_loss_fn_name} should be retrievable from registry",
                )
                self.assertTrue(
                    issubclass(entropy_loss_fn_cls, EntropyLossFn),
                    f"{entropy_loss_fn_name} should be a subclass of EntropyLossFn",
                )
        with self.assertRaises(ValueError):
            ENTROPY_LOSS_FN.get("non_existent_entropy_loss_fn")

        # test kl function
        kl_fn_names = list(KL_FN._default_mapping.keys())
        for kl_fn_name in kl_fn_names:
            with self.subTest(kl_fn_name=kl_fn_name):
                kl_fn_cls = KL_FN.get(kl_fn_name)
                self.assertIsNotNone(kl_fn_cls, f"{kl_fn_name} should be retrievable from registry")
                self.assertTrue(
                    issubclass(kl_fn_cls, KLFn), f"{kl_fn_name} should be a subclass of KLFn"
                )
        with self.assertRaises(ValueError):
            KL_FN.get("non_existent_kl_fn")

        # test policy loss function
        policy_loss_fn_names = list(POLICY_LOSS_FN._default_mapping.keys())
        for policy_loss_fn_name in policy_loss_fn_names:
            with self.subTest(policy_loss_fn_name=policy_loss_fn_name):
                policy_loss_fn_cls = POLICY_LOSS_FN.get(policy_loss_fn_name)
                self.assertIsNotNone(
                    policy_loss_fn_cls, f"{policy_loss_fn_name} should be retrievable from registry"
                )
                self.assertTrue(
                    issubclass(policy_loss_fn_cls, PolicyLossFn),
                    f"{policy_loss_fn_name} should be a subclass of PolicyLossFn",
                )
        with self.assertRaises(ValueError):
            POLICY_LOSS_FN.get("non_existent_policy_loss_fn")

        # test sample strategy
        sample_strategy_names = list(SAMPLE_STRATEGY._default_mapping.keys())
        for sample_strategy_name in sample_strategy_names:
            with self.subTest(sample_strategy_name=sample_strategy_name):
                sample_strategy_cls = SAMPLE_STRATEGY.get(sample_strategy_name)
                self.assertIsNotNone(
                    sample_strategy_cls,
                    f"{sample_strategy_name} should be retrievable from registry",
                )
                self.assertTrue(
                    issubclass(sample_strategy_cls, SampleStrategy),
                    f"{sample_strategy_name} should be a subclass of SampleStrategy",
                )
        with self.assertRaises(ValueError):
            SAMPLE_STRATEGY.get("non_existent_sample_strategy")

    def test_buffer_module_registry_mapping(self):
        """Test registry mapping in buffer module"""
        # test experience operator
        operator_names = list(EXPERIENCE_OPERATORS._default_mapping.keys())
        for operator_name in operator_names:
            with self.subTest(operator_name=operator_name):
                operator_cls = EXPERIENCE_OPERATORS.get(operator_name)
                self.assertIsNotNone(
                    operator_cls, f"{operator_name} should be retrievable from registry"
                )
                self.assertTrue(
                    issubclass(operator_cls, ExperienceOperator),
                    f"{operator_name} should be a subclass of ExperienceOperator",
                )
        with self.assertRaises(ValueError):
            EXPERIENCE_OPERATORS.get("non_existent_operator")

        # test reader
        reader_names = list(READER._default_mapping.keys())
        for reader_name in reader_names:
            with self.subTest(reader_name=reader_name):
                reader_cls = READER.get(reader_name)
                self.assertIsNotNone(
                    reader_cls, f"{reader_name} should be retrievable from registry"
                )
                self.assertTrue(
                    issubclass(reader_cls, BufferReader),
                    f"{reader_name} should be a subclass of BufferReader",
                )
        with self.assertRaises(ValueError):
            READER.get("non_existent_reader")

        # test formatter
        formatter_names = list(FORMATTER._default_mapping.keys())
        for formatter_name in formatter_names:
            with self.subTest(formatter_name=formatter_name):
                formatter_cls = FORMATTER.get(formatter_name)
                self.assertIsNotNone(
                    formatter_cls, f"{formatter_name} should be retrievable from registry"
                )
        with self.assertRaises(ValueError):
            FORMATTER.get("non_existent_formatter")

        # test sql schema
        schema_names = list(SQL_SCHEMA._default_mapping.keys())
        for schema_name in schema_names:
            with self.subTest(schema_name=schema_name):
                schema_cls = SQL_SCHEMA.get(schema_name)
                self.assertIsNotNone(
                    schema_cls, f"{schema_name} should be retrievable from registry"
                )
        with self.assertRaises(ValueError):
            SQL_SCHEMA.get("non_existent_schema")

        # test selector
        selector_names = list(SELECTORS._default_mapping.keys())
        for selector_name in selector_names:
            with self.subTest(selector_name=selector_name):
                selector_cls = SELECTORS.get(selector_name)
                self.assertIsNotNone(
                    selector_cls, f"{selector_name} should be retrievable from registry"
                )
                self.assertTrue(
                    issubclass(selector_cls, BaseSelector),
                    f"{selector_name} should be a subclass of BaseSelector",
                )
        with self.assertRaises(ValueError):
            SELECTORS.get("non_existent_selector")

        # test priority function
        priority_fn_names = list(PRIORITY_FUNC._default_mapping.keys())

        for priority_fn_name in priority_fn_names:
            with self.subTest(priority_fn_name=priority_fn_name):
                priority_fn_cls = PRIORITY_FUNC.get(priority_fn_name)
                self.assertIsNotNone(
                    priority_fn_cls, f"{priority_fn_name} should be retrievable from registry"
                )
                self.assertTrue(
                    issubclass(priority_fn_cls, PriorityFunction),
                    f"{priority_fn_name} should be a subclass of PriorityFunction",
                )
        with self.assertRaises(ValueError):
            PRIORITY_FUNC.get("non_existent_priority_fn")

    def test_utils_module_registry_mapping(self):
        """Test registry mapping in utils module"""
        # test monitor
        monitor_names = list(MONITOR._default_mapping.keys())
        for monitor_name in monitor_names:
            with self.subTest(monitor_name=monitor_name):
                monitor_cls = MONITOR.get(monitor_name)
                self.assertIsNotNone(
                    monitor_cls, f"{monitor_name} should be retrievable from registry"
                )
                self.assertTrue(
                    issubclass(monitor_cls, Monitor),
                    f"{monitor_name} should be a subclass of Monitor",
                )
        with self.assertRaises(ValueError):
            MONITOR.get("non_existent_monitor")

    def test_register_module(self):
        """Test register module functionality"""
        # Test that the registered class can be retrieved from registry
        retrieved_cls = ENTROPY_LOSS_FN.get("dummy_entropy_loss_fn")
        self.assertIsNotNone(
            retrieved_cls, "dummy_entropy_loss_fn should be retrievable from registry"
        )
        self.assertTrue(
            issubclass(retrieved_cls, EntropyLossFn),
            "dummy_entropy_loss_fn should be a subclass of EntropyLossFn",
        )
        self.assertEqual(
            retrieved_cls, DummyEntropyLossFn, "Retrieved class should be DummyEntropyLossFn"
        )

        # Test that the registered class can be instantiated and used
        instance = retrieved_cls(entropy_coef=0.1)
        self.assertIsInstance(instance, EntropyLossFn)
        self.assertEqual(instance.entropy_coef, 0.1)

        # Test that the instance can be called (basic functionality)
        loss, metrics = instance(entropy=torch.tensor(1.0), action_mask=torch.tensor(1.0))
        self.assertEqual(loss.item(), 0.0)
        self.assertIsInstance(metrics, dict)
