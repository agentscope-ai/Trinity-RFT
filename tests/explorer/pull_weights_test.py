"""Unit tests for Explorer._pull_latest_weights recovery logic."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from parameterized import parameterized

from trinity.explorer.explorer import Explorer


class TestPullLatestWeights(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.explorer = object.__new__(Explorer)
        self.explorer.logger = MagicMock()
        self.explorer.models = [MagicMock(), MagicMock()]
        self.explorer.synchronizer = MagicMock()

    def _setup_versions(self, model_version: int, new_version: int):
        self.explorer.model_version = model_version
        self.explorer.synchronizer.wait_new_model_state_dict.remote = AsyncMock(
            return_value=new_version,
        )
        for m in self.explorer.models:
            m.sync_model.remote = AsyncMock()

    @parameterized.expand(
        [
            # (model_version, new_version, expect_sync)
            (-1, 0, False),  # fresh start: version 0 = base model, no sync needed
            (-1, 3, True),  # recovery: trainer already trained, must sync
            (2, 4, True),  # normal periodic sync
            (3, 3, False),  # no new version available
        ]
    )
    async def test_pull_latest_weights(self, model_version, new_version, expect_sync):
        self._setup_versions(model_version, new_version)

        await Explorer._pull_latest_weights(self.explorer)

        expected_version = max(model_version, new_version)
        self.assertEqual(self.explorer.model_version, expected_version)

        for m in self.explorer.models:
            if expect_sync:
                m.sync_model.remote.assert_called_once_with(new_version)
            else:
                m.sync_model.remote.assert_not_called()

    async def test_no_new_version_logs_warning(self):
        self._setup_versions(model_version=3, new_version=3)

        await Explorer._pull_latest_weights(self.explorer)

        self.explorer.logger.warning.assert_called_once()
