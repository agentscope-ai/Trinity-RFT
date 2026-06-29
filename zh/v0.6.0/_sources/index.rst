.. Trinity-RFT documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎来到 Trinity-RFT 的文档!
=======================================

.. include:: main.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 2
   :glob:
   :hidden:
   :caption: 教程

   tutorial/trinity_installation.md
   tutorial/develop_overview.md
   tutorial/develop_workflow.md
   tutorial/develop_algorithm.md
   tutorial/example_mix_algo.md
   tutorial/develop_operator.md
   tutorial/develop_selector.md
   tutorial/trinity_configs.md
   tutorial/trinity_gpu_configs.md
   tutorial/synchronizer.md
   tutorial/align_with_verl.md
   tutorial/metrics_reference.md

.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:
   :caption: 样例

   tutorial/example_reasoning_basic.md
   tutorial/example_reasoning_advanced.md
   tutorial/example_async_mode.md
   tutorial/example_multi_turn.md
   tutorial/example_step_wise.md
   tutorial/example_react.md
   tutorial/example_search_email.md
   tutorial/example_dpo.md
   tutorial/example_tinker_backend.md
   tutorial/example_megatron.md
   tutorial/example_data_functionalities.md
   tutorial/example_dataset_perspective.md

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 常见问题

   tutorial/faq.md

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api_reference
