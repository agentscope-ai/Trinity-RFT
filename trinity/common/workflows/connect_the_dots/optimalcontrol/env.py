# -*- coding: utf-8 -*-
"""Optimal-control environment with hidden linear dynamics.

State: s_t = (x_t, v_t)
Dynamics:
    v_{t+1} = a_env * v_t + b_env * u_t + epsilon_t
    x_{t+1} = x_t + v_{t+1}
Control constraint: u_t in [-1, 1]
"""

import ast
import math
import operator
import random
from typing import Any, Callable, Dict, List, Tuple, TypedDict
from xml.sax.saxutils import escape

ActionFn = Callable[[float, float, int, float, float], float]


class ResolvedOptimalControlTask(TypedDict):
    a_env: float
    b_env: float
    min_abs_b_env: float  #!!!
    x0: float
    v0: float
    x_target: float
    v_target: float
    horizon: int
    control_penalty_coef: float
    enable_process_noise: bool  #!!!
    process_noise_std: float  #!!!
    task_seed: int  #!!!


def _clip(value: float, lower: float, upper: float) -> float:
    """Clip value to [lower, upper]."""
    return max(lower, min(value, upper))


def format_reward(value: float) -> str:
    """Keep three decimals unless a positive reward would appear to be zero."""
    rounded = f"{value:.3f}"
    if value > 0.0 and rounded == "0.000":
        return f"{value:.3e} (nonzero; below three-decimal precision)"
    return rounded


_VARIABLE_NAMES = {
    "x",
    "v",
    "t",
    "x_target",
    "v_target",
    "horizon",
    "remaining_steps",
}
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}
_COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_FUNCTION_COMPARISONS = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
}
_FUNCTION_ARITY = {
    "abs": (1, 1),
    "min": (1, None),
    "max": (1, None),
    "pow": (2, 2),
    "clip": (3, 3),
    "ifelse": (3, 3),
    "lt": (2, 2),
    "le": (2, 2),
    "gt": (2, 2),
    "ge": (2, 2),
    "eq": (2, 2),
    "ne": (2, 2),
}


def _error_detail(error: Exception) -> str:
    """Return a concise error description suitable for model feedback."""
    return f"{type(error).__name__}: {error}"[:500]


def _validate_expression_node(node: ast.AST) -> None:
    """Validate the supported scalar expression language."""
    if isinstance(node, ast.Expression):
        _validate_expression_node(node.body)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
        try:
            finite = math.isfinite(float(node.value))
        except OverflowError as error:
            raise ValueError("numeric constants must be finite") from error
        if not finite:
            raise ValueError("numeric constants must be finite")
        return
    if isinstance(node, ast.Name):
        if node.id not in _VARIABLE_NAMES:
            raise ValueError(f"unknown variable: {node.id}")
        return
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY_OPERATORS:
            raise ValueError("unsupported binary operator")
        _validate_expression_node(node.left)
        _validate_expression_node(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPERATORS:
            raise ValueError("unsupported unary operator")
        _validate_expression_node(node.operand)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTION_ARITY:
            raise ValueError("unsupported function call")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        minimum, maximum = _FUNCTION_ARITY[node.func.id]
        if len(node.args) < minimum or (maximum is not None and len(node.args) > maximum):
            raise ValueError(f"invalid argument count for {node.func.id}")
        for argument in node.args:
            _validate_expression_node(argument)
        return
    if isinstance(node, ast.IfExp):
        _validate_expression_node(node.test)
        _validate_expression_node(node.body)
        _validate_expression_node(node.orelse)
        return
    if isinstance(node, ast.Compare):
        if any(type(comparator) not in _COMPARISON_OPERATORS for comparator in node.ops):
            raise ValueError("unsupported comparison operator")
        _validate_expression_node(node.left)
        for comparator in node.comparators:
            _validate_expression_node(comparator)
        return
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError("unsupported boolean operator")
        for value in node.values:
            _validate_expression_node(value)
        return
    raise ValueError(f"unsupported expression syntax: {type(node).__name__}")


def validate_action_expression(expression: object) -> str:
    """Parse and validate one controller expression."""
    if not isinstance(expression, str):
        raise ValueError("action_expression must be text")
    expression = expression.strip()
    if not expression:
        raise ValueError("action_expression cannot be empty")
    try:
        tree = ast.parse(expression, mode="eval")
        _validate_expression_node(tree)
    except SyntaxError as error:
        raise ValueError(f"invalid action_expression syntax: {error.msg}") from error
    except RecursionError as error:
        raise ValueError("action_expression is too deeply nested") from error
    return expression


def _evaluate_expression(node: ast.AST, variables: Dict[str, float]) -> Any:
    """Evaluate one previously validated expression node."""
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return variables[node.id]
    if isinstance(node, ast.BinOp):
        return _BINARY_OPERATORS[type(node.op)](
            _evaluate_expression(node.left, variables),
            _evaluate_expression(node.right, variables),
        )
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPERATORS[type(node.op)](_evaluate_expression(node.operand, variables))
    if isinstance(node, ast.Call):
        function_name = node.func.id
        if function_name == "ifelse":
            condition = _evaluate_expression(node.args[0], variables)
            branch = node.args[1] if condition else node.args[2]
            return _evaluate_expression(branch, variables)
        values = [_evaluate_expression(argument, variables) for argument in node.args]
        if function_name == "abs":
            return abs(values[0])
        if function_name == "min":
            return min(values)
        if function_name == "max":
            return max(values)
        if function_name == "pow":
            return pow(values[0], values[1])
        if function_name == "clip":
            return _clip(values[0], values[1], values[2])
        return _FUNCTION_COMPARISONS[function_name](values[0], values[1])
    if isinstance(node, ast.IfExp):
        branch = node.body if _evaluate_expression(node.test, variables) else node.orelse
        return _evaluate_expression(branch, variables)
    if isinstance(node, ast.Compare):
        left = _evaluate_expression(node.left, variables)
        for operation, comparator in zip(node.ops, node.comparators):
            right = _evaluate_expression(comparator, variables)
            if not _COMPARISON_OPERATORS[type(operation)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = _evaluate_expression(node.values[0], variables)
            for value in node.values[1:]:
                if not result:
                    return result
                result = _evaluate_expression(value, variables)
            return result
        result = _evaluate_expression(node.values[0], variables)
        for value in node.values[1:]:
            if result:
                return result
            result = _evaluate_expression(value, variables)
        return result
    raise TypeError(f"unsupported expression node: {type(node).__name__}")


def compile_action_expression(expression: str, horizon: int) -> Tuple[ActionFn | None, str | None]:
    """Compile one validated scalar expression into a feedback action."""
    try:
        expression = validate_action_expression(expression)
        tree = ast.parse(expression, mode="eval")
    except ValueError as error:
        return None, str(error)

    def action(x, v, t, x_target, v_target):
        variables = {
            "x": float(x),
            "v": float(v),
            "t": float(t),
            "x_target": float(x_target),
            "v_target": float(v_target),
            "horizon": float(horizon),
            "remaining_steps": float(max(1, horizon - t)),
        }
        return _evaluate_expression(tree.body, variables)

    return action, None


def serialize_action_expression_xml(expression: str) -> str:
    """Serialize a validated expression as canonical controller XML."""
    expression = validate_action_expression(expression)
    return (
        "<answer>\n"
        "  <action_expression>\n"
        f"    {escape(expression)}\n"
        "  </action_expression>\n"
        "</answer>"
    )


class OptimalControlEnv:
    """One-dimensional optimal-control environment used by CoD-Deploy demos."""

    @staticmethod  #!!!
    def _sample_b_env(  #!!!
        rng: random.Random, lower: float, upper: float, min_abs: float  #!!!
    ) -> float:  #!!!
        """Sample uniformly while excluding (-min_abs, min_abs)."""  #!!!
        if lower > upper:  #!!!
            raise ValueError("b_env_range lower bound must not exceed upper bound")  #!!!
        if min_abs < 0.0:  #!!!
            raise ValueError("min_abs_b_env must be non-negative")  #!!!
        if min_abs == 0.0:  #!!!
            return rng.uniform(lower, upper)  #!!!
        left_upper = min(upper, -min_abs)  #!!!
        right_lower = max(lower, min_abs)  #!!!
        left_length = max(0.0, left_upper - lower)  #!!!
        right_length = max(0.0, upper - right_lower)  #!!!
        total_length = left_length + right_length  #!!!
        if total_length == 0.0:  #!!!
            if lower <= -min_abs <= upper:  #!!!
                return -min_abs  #!!!
            if lower <= min_abs <= upper:  #!!!
                return min_abs  #!!!
            raise ValueError("b_env_range contains no value satisfying min_abs_b_env")  #!!!
        offset = rng.random() * total_length  #!!!
        if offset < left_length:  #!!!
            return lower + offset  #!!!
        return right_lower + (offset - left_length)  #!!!

    @staticmethod
    def resolve_task(
        raw_task: Dict[str, Any],
        workflow_args: Dict[str, Any],
    ) -> ResolvedOptimalControlTask:
        """Resolve one task against its runtime CoD pack environment."""
        environment_seed = int(raw_task.get("pack_seed", raw_task["seed"]))
        rng = random.Random(environment_seed)
        a_env = rng.uniform(*workflow_args["a_env_range"])
        b_lower, b_upper = map(float, workflow_args["b_env_range"])  #!!!
        min_abs_b_env = float(workflow_args.get("min_abs_b_env", 0.0))  #!!!
        b_env = OptimalControlEnv._sample_b_env(  #!!!
            rng, b_lower, b_upper, min_abs_b_env  #!!!
        )  #!!!
        x0 = float(raw_task["x0"])
        v0 = float(raw_task["v0"])
        x_target = float(raw_task["x_target"])
        v_target = float(raw_task["v_target"])
        horizon_value = raw_task.get("horizon")  #!!!
        if horizon_value is None:  #!!!
            horizon_value = raw_task["max_horizon"]  #!!!
        horizon = int(horizon_value)  #!!!
        control_penalty_coef = float(raw_task["control_penalty_coef"])
        enable_process_noise = bool(  #!!!
            workflow_args.get("enable_process_noise", False)  #!!!
        )  #!!!
        process_noise_std = float(workflow_args.get("process_noise_std", 0.0))  #!!!
        if process_noise_std < 0.0:  #!!!
            raise ValueError("process_noise_std must be non-negative")  #!!!
        task_seed = int(raw_task["seed"])  #!!!

        return {
            "a_env": a_env,
            "b_env": b_env,
            "min_abs_b_env": min_abs_b_env,  #!!!
            "x0": x0,
            "v0": v0,
            "x_target": x_target,
            "v_target": v_target,
            "horizon": horizon,
            "control_penalty_coef": control_penalty_coef,
            "enable_process_noise": enable_process_noise,  #!!!
            "process_noise_std": process_noise_std,  #!!!
            "task_seed": task_seed,  #!!!
        }

    def __init__(
        self,
        a_env: float,
        b_env: float,
        x0: float,
        v0: float,
        x_target: float,
        v_target: float,
        horizon: int = 8,
        control_penalty_coef: float = 0.03,
        enable_process_noise: bool = False,  #!!!
        process_noise_std: float = 0.0,  #!!!
        task_seed: int = 0,  #!!!
    ):
        self.a_env = float(a_env)
        self.b_env = float(b_env)
        self.x0 = float(x0)
        self.v0 = float(v0)
        self.x_target = float(x_target)
        self.v_target = float(v_target)
        self.horizon = int(horizon)
        self.control_penalty_coef = float(control_penalty_coef)
        self.enable_process_noise = bool(enable_process_noise)  #!!!
        self.process_noise_std = float(process_noise_std)  #!!!
        if self.process_noise_std < 0.0:  #!!!
            raise ValueError("process_noise_std must be non-negative")  #!!!
        self.task_seed = int(task_seed)  #!!!

    def reset(self) -> Tuple[float, float]:
        """Reset to initial state."""
        return self.x0, self.v0

    def step(  #!!!
        self, x: float, v: float, u: float, process_noise: float = 0.0  #!!!
    ) -> Tuple[float, float]:  #!!!
        """Apply one control and return the next state."""
        u = float(_clip(u, -1.0, 1.0))
        v_next = self.a_env * v + self.b_env * u + float(process_noise)  #!!!
        x_next = x + v_next
        return x_next, v_next

    def rollout(self, action_fn: ActionFn) -> Dict[str, Any]:
        """Roll out the policy for the full horizon.

        Args:
            action_fn: callable action(x, v, t, x_target, v_target) -> u_t.

        Returns:
            dict with rollout states, controls, terminal metrics, and error status.  #!!!
        """
        x, v = self.reset()
        xs = [x]
        vs = [v]
        us: List[float] = []
        process_noises: List[float] = []  #!!!
        noise_rng = random.Random(self.task_seed ^ 0x5DEECE66D)  #!!!

        for t in range(self.horizon):
            try:
                raw_u = action_fn(x, v, t, self.x_target, self.v_target)  #!!!
                if isinstance(raw_u, bool) or not isinstance(raw_u, (int, float)):  #!!!
                    raise TypeError("action must return an int or float scalar")  #!!!
                u = float(raw_u)  #!!!
                if not math.isfinite(u):
                    raise ValueError("action must return a finite scalar")
                u = _clip(u, -1.0, 1.0)
            except Exception as error:
                raise RuntimeError(f"action failed at t={t}: {_error_detail(error)}") from error
            us.append(u)
            process_noise = (  #!!!
                noise_rng.gauss(0.0, self.process_noise_std)  #!!!
                if self.enable_process_noise and self.process_noise_std > 0.0  #!!!
                else 0.0  #!!!
            )  #!!!
            process_noises.append(process_noise)  #!!!
            x, v = self.step(x, v, u, process_noise=process_noise)  #!!!
            xs.append(x)
            vs.append(v)

        loss = self.compute_loss(xs[-1], vs[-1], us)  #!!!
        reward = self.compute_reward(loss)  #!!!
        process_noise_rms = (  #!!!
            math.sqrt(  #!!!
                sum(noise * noise for noise in process_noises) / len(process_noises)  #!!!
            )  #!!!
            if process_noises  #!!!
            else 0.0  #!!!
        )  #!!!

        return {
            "xs": xs,
            "vs": vs,
            "us": us,
            "process_noises": process_noises,  #!!!
            "process_noise_rms": process_noise_rms,  #!!!
            "x_final": xs[-1],
            "v_final": vs[-1],
            "loss": loss,
            "reward": reward,
        }

    def rollout_action_expression(
        self,
        expression: str,
    ) -> Tuple[Dict[str, Any] | None, str, str | None]:
        """Compile and fully evaluate one feedback expression."""
        action_fn, validation_error = compile_action_expression(expression, self.horizon)
        if action_fn is None:
            return None, "format_error", validation_error
        try:
            return self.rollout(action_fn), "ok", None
        except Exception as error:
            return None, "action_error", _error_detail(error)

    def compute_loss(  #!!!
        self, x_final: float, v_final: float, us: List[float]  #!!!
    ) -> float:  #!!!
        """Compute terminal-state and control-effort loss."""  #!!!
        pos_error = x_final - self.x_target  #!!!
        vel_error = v_final - self.v_target  #!!!
        control_cost = self.control_penalty_coef * sum(u * u for u in us)  #!!!
        return pos_error * pos_error + 2.0 * vel_error * vel_error + control_cost  #!!!

    def compute_reward(self, loss: float) -> float:
        """Convert loss to reward."""
        return 1.0 / (1.0 + loss)

    def render_trajectory(self, result: Dict[str, Any]) -> str:
        """Render rollout result as a table aligned by time step.

        The control column u_t shows the control applied *at* step t while the
        system is in state (x_t, v_t). At the terminal step T no further control
        is available, so it is marked as ``--``.
        """
        xs = result["xs"]
        vs = result["vs"]
        us = result["us"]
        lines = [
            f"{'t':>3}  {'x_t':>10}  {'v_t':>10}  {'u_t':>10}",
            f"{0:>3}  {xs[0]:>10.3f}  {vs[0]:>10.3f}  {us[0]:>10.3f}",
        ]
        for t in range(1, self.horizon):
            u_str = f"{us[t]:>10.3f}"
            lines.append(f"{t:>3}  {xs[t]:>10.3f}  {vs[t]:>10.3f}  {u_str}")
        lines.append(
            f"{self.horizon:>3}  {xs[self.horizon]:>10.3f}  {vs[self.horizon]:>10.3f}  {'--':>10}"
        )
        return "\n".join(lines)

    def render_summary(self, result: Dict[str, Any]) -> str:
        """Render the terminal state, target, loss, and reward."""
        reward_text = format_reward(float(result["reward"]))
        lines = [
            "Terminal state:",
            f"x_{self.horizon} = {result['x_final']:.3f}, v_{self.horizon} = {result['v_final']:.3f}.",
            "Target:",
            f"x_target = {self.x_target:.3f}, v_target = {self.v_target:.3f}.",
            "Loss and reward:",
            f"L = {result['loss']:.3f}, R = {reward_text}.",
        ]
        return "\n".join(lines)
