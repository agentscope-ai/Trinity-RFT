from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn, GroupAdvantage
from trinity.utils.registry import Registry

ADVANTAGE_FN: Registry = Registry(
    "advantage_fn",
    default_mapping={
        "ppo": "trinity.algorithm.advantage_fn.ppo_advantage.PPOAdvantageFn",
        "grpo": "trinity.algorithm.advantage_fn.grpo_advantage.GRPOGroupedAdvantage",
        "grpo_verl": "trinity.algorithm.advantage_fn.grpo_advantage.GRPOAdvantageFn",
        "step_wise_grpo": "trinity.algorithm.advantage_fn.multi_step_grpo_advantage.StepWiseGRPOAdvantageFn",
        "reinforceplusplus": "trinity.algorithm.advantage_fn.reinforce_plus_plus_advantage.REINFORCEPLUSPLUSAdvantageFn",
        "reinforce": "trinity.algorithm.advantage_fn.reinforce_advantage.REINFORCEGroupAdvantage",
        "remax": "trinity.algorithm.advantage_fn.remax_advantage.REMAXAdvantageFn",
        "rloo": "trinity.algorithm.advantage_fn.rloo_advantage.RLOOAdvantageFn",
        "opmd": "trinity.algorithm.advantage_fn.opmd_advantage.OPMDGroupAdvantage",
        "opmd_verl": "trinity.algorithm.advantage_fn.opmd_advantage.OPMDAdvantageFn",
        "asymre": "trinity.algorithm.advantage_fn.asymre_advantage.ASYMREGroupAdvantage",
        "asymre_verl": "trinity.algorithm.advantage_fn.asymre_advantage.ASYMREAdvantageFn",
        "rec": "trinity.algorithm.advantage_fn.rec_advantage.RECGroupedAdvantage",
        "on_policy_distill": "trinity.algorithm.advantage_fn.on_policy_distill_advantage.OnPolicyDistillAdvantage",
        "jsd": "trinity.algorithm.advantage_fn.jsd_advantage.JSDAdvantage",
        "clipb": "trinity.algorithm.advantage_fn.clipb_advantage.ClipBAdvantageFn",
    },
)

__all__ = [
    "ADVANTAGE_FN",
    "AdvantageFn",
    "GroupAdvantage",
]
