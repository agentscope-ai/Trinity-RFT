from trinity.algorithm.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.algorithm.algorithm import AlgorithmType
from trinity.algorithm.entropy_loss_fn import ENTROPY_LOSS_FN, EntropyLossFn
from trinity.algorithm.kl_fn import KL_FN, KLFn
from trinity.algorithm.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.sample_strategy import SAMPLE_STRATEGY, SampleStrategy
from trinity.utils.registry import Registry

ALGORITHM_TYPE: Registry = Registry(
    "algorithm",
    default_mapping={
        "sft": "trinity.algorithm.algorithm.SFTAlgorithm",
        "ppo": "trinity.algorithm.algorithm.PPOAlgorithm",
        "grpo": "trinity.algorithm.algorithm.GRPOAlgorithm",
        "reinforceplusplus": "trinity.algorithm.algorithm.ReinforcePlusPlusAlgorithm",
        "rloo": "trinity.algorithm.algorithm.RLOOAlgorithm",
        "opmd": "trinity.algorithm.algorithm.OPMDAlgorithm",
        "asymre": "trinity.algorithm.algorithm.AsymREAlgorithm",
        "dpo": "trinity.algorithm.algorithm.DPOAlgorithm",
        "topr": "trinity.algorithm.algorithm.TOPRAlgorithm",
        "cispo": "trinity.algorithm.algorithm.CISPOAlgorithm",
        "gspo": "trinity.algorithm.algorithm.GSPOAlgorithm",
        "sapo": "trinity.algorithm.algorithm.SAPOAlgorithm",
        "mix": "trinity.algorithm.algorithm.MIXAlgorithm",
        "mix_chord": "trinity.algorithm.algorithm.MIXCHORDAlgorithm",
        "raft": "trinity.algorithm.algorithm.RAFTAlgorithm",
        "sppo": "trinity.algorithm.algorithm.sPPOAlgorithm",
        "rec": "trinity.algorithm.algorithm.RECAlgorithm",
        "multi_step_grpo": "trinity.algorithm.algorithm.MultiStepGRPOAlgorithm",
        "on_policy_distill": "trinity.algorithm.algorithm.OnPolicyDistillAlgorithm",
        "jsd": "trinity.algorithm.algorithm.JSDAlgorithm",
        "grpo_verl": "trinity.algorithm.algorithm.GRPOverlAlgorithm",
    },
)

__all__ = [
    "ALGORITHM_TYPE",
    "AlgorithmType",
    "AdvantageFn",
    "ADVANTAGE_FN",
    "PolicyLossFn",
    "POLICY_LOSS_FN",
    "KLFn",
    "KL_FN",
    "EntropyLossFn",
    "ENTROPY_LOSS_FN",
    "SampleStrategy",
    "SAMPLE_STRATEGY",
]
