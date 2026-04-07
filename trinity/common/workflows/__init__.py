# -*- coding: utf-8 -*-
"""Workflow module"""
from trinity.common.workflows.workflow import Task, Workflow
from trinity.utils.registry import Registry

WORKFLOWS: Registry = Registry(
    "workflows",
    default_mapping={
        # simple/math
        "simple_workflow": "trinity.common.workflows.workflow.SimpleWorkflow",
        "async_simple_workflow": "trinity.common.workflows.workflow.AsyncSimpleWorkflow",
        "math_workflow": "trinity.common.workflows.workflow.MathWorkflow",
        "async_math_workflow": "trinity.common.workflows.workflow.AsyncMathWorkflow",
        "math_boxed_workflow": "trinity.common.workflows.customized_math_workflows.MathBoxedWorkflow",
        "async_math_boxed_workflow": "trinity.common.workflows.customized_math_workflows.AsyncMathBoxedWorkflow",
        "math_eval_workflow": "trinity.common.workflows.eval_workflow.MathEvalWorkflow",
        "async_math_eval_workflow": "trinity.common.workflows.eval_workflow.AsyncMathEvalWorkflow",
        "math_rm_workflow": "trinity.common.workflows.math_rm_workflow.MathRMWorkflow",
        "async_math_rm_workflow": "trinity.common.workflows.math_rm_workflow.AsyncMathRMWorkflow",
        # tool_call
        "tool_call_workflow": "trinity.common.workflows.customized_toolcall_workflows.ToolCallWorkflow",
        # agentscope
        "agentscope_workflow_adapter": "trinity.common.workflows.agentscope_workflow.AgentScopeWorkflowAdapter",
        "agentscope_workflow_adapter_v1": "trinity.common.workflows.agentscope_workflow.AgentScopeWorkflowAdapterV1",
        "agentscope_react_workflow": "trinity.common.workflows.agentscope.react.react_workflow.AgentScopeReActWorkflow",
        "agentscope_react_math_workflow": "trinity.common.workflows.envs.agentscope.agentscopev1_react_workflow.AgentScopeReactMathWorkflow",
        "as_react_workflow": "trinity.common.workflows.agentscope.react.react_workflow.AgentScopeReActWorkflow",
        "agentscopev0_react_math_workflow": "trinity.common.workflows.envs.agentscope.agentscopev0_react_workflow.AgentScopeV0ReactMathWorkflow",
        "agentscope_v1_react_search_workflow": "trinity.common.workflows.envs.agentscope.agentscopev1_search_workflow.AgentScopeV1ReactSearchWorkflow",
        "email_search_workflow": "trinity.common.workflows.envs.email_searcher.workflow.EmailSearchWorkflow",
        # concatenated multi-turn
        "alfworld_workflow": "trinity.common.workflows.envs.alfworld.alfworld_workflow.AlfworldWorkflow",
        "RAFT_alfworld_workflow": "trinity.common.workflows.envs.alfworld.RAFT_alfworld_workflow.RAFTAlfworldWorkflow",
        "RAFT_reflect_alfworld_workflow": "trinity.common.workflows.envs.alfworld.RAFT_reflect_alfworld_workflow.RAFTReflectAlfworldWorkflow",
        "frozen_lake_workflow": "trinity.common.workflows.envs.frozen_lake.workflow.FrozenLakeWorkflow",
        "sciworld_workflow": "trinity.common.workflows.envs.sciworld.sciworld_workflow.SciWorldWorkflow",
        "webshop_workflow": "trinity.common.workflows.envs.webshop.webshop_workflow.WebShopWorkflow",
        # general multi-turn
        "step_wise_alfworld_workflow": "trinity.common.workflows.envs.alfworld.alfworld_workflow.StepWiseAlfworldWorkflow",
        # ruler/llm_as_a_judge
        "async_math_ruler_workflow": "trinity.common.workflows.math_ruler_workflow.AsyncMathRULERWorkflow",
        "math_ruler_workflow": "trinity.common.workflows.math_ruler_workflow.MathRULERWorkflow",
        "math_trainable_ruler_workflow": "trinity.common.workflows.math_trainable_ruler_workflow.MathTrainableRULERWorkflow",
        "rubric_judge_workflow": "trinity.common.workflows.rubric_judge_workflow.RubricJudgeWorkflow",
        "rubric_judge_openai_workflow": "trinity.common.workflows.rubric_judge_workflow.RubricJudgeWorkflowWithAPI",
        # others
        "simple_mm_workflow": "trinity.common.workflows.simple_mm_workflow.SimpleMMWorkflow",
        "async_simple_mm_workflow": "trinity.common.workflows.simple_mm_workflow.AsyncSimpleMMWorkflow",
        # on-policy distillation workflows
        "on_policy_distill_workflow": "trinity.common.workflows.on_policy_distill_workflow.OnPolicyDistillWorkflow",
        "on_policy_distill_math_workflow": "trinity.common.workflows.on_policy_distill_workflow.OnPolicyDistillMathWorkflow",
        # custom workflows
        "sudoku_workflow": "trinity.common.workflows.envs.sudoku.sudoku_workflow.SudokuWorkflow",
    },
)

__all__ = [
    "Task",
    "Workflow",
    "WORKFLOWS",
]
