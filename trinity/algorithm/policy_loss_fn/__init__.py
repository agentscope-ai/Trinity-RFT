from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.utils.registry import Registry

POLICY_LOSS_FN: Registry = Registry(
    "policy_loss_fn",
    default_mapping={
        "ppo": "trinity.algorithm.policy_loss_fn.ppo_policy_loss.PPOPolicyLossFn",
        "opmd": "trinity.algorithm.policy_loss_fn.opmd_policy_loss.OPMDPolicyLossFn",
        "dpo": "trinity.algorithm.policy_loss_fn.dpo_loss.DPOLossFn",
        "sft": "trinity.algorithm.policy_loss_fn.sft_loss.SFTLossFn",
        "mix": "trinity.algorithm.policy_loss_fn.mix_policy_loss.MIXPolicyLossFn",
        "gspo": "trinity.algorithm.policy_loss_fn.gspo_policy_loss.GSPOLossFn",
        "topr": "trinity.algorithm.policy_loss_fn.topr_policy_loss.TOPRPolicyLossFn",
        "cispo": "trinity.algorithm.policy_loss_fn.cispo_policy_loss.CISPOPolicyLossFn",
        "sft_is": "trinity.algorithm.policy_loss_fn.chord_policy_loss.SFTISLossFn",
        "sft_phi": "trinity.algorithm.policy_loss_fn.chord_policy_loss.SFTPhiLossFn",
        "mix_chord": "trinity.algorithm.policy_loss_fn.chord_policy_loss.MIXCHORDPolicyLossFn",
        "sppo": "trinity.algorithm.policy_loss_fn.sppo_loss_fn.sPPOPolicyLossFn",
        "rec": "trinity.algorithm.policy_loss_fn.rec_policy_loss.RECPolicyLossFn",
        "sapo": "trinity.algorithm.policy_loss_fn.sapo_policy_loss.SAPOPolicyLossFn",
        "importance_sampling": "trinity.algorithm.policy_loss_fn.importance_sampling_policy_loss.ImportanceSamplingLossFn",
        "m2po": "trinity.algorithm.policy_loss_fn.m2po_policy_loss.M2POPolicyLossFn",
    },
)

__all__ = [
    "POLICY_LOSS_FN",
    "PolicyLossFn",
]
