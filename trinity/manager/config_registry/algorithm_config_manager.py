import streamlit as st

from trinity.algorithm import ALGORITHM_TYPE
from trinity.algorithm.advantage_fn import ADVANTAGE_FN
from trinity.algorithm.advantage_fn.grpo_advantage import GRPOAdvantageFn
from trinity.algorithm.advantage_fn.opmd_advantage import OPMDAdvantageFn
from trinity.algorithm.advantage_fn.ppo_advantage import PPOAdvantageFn
from trinity.algorithm.algorithm import GRPOAlgorithm
from trinity.algorithm.entropy_loss_fn import ENTROPY_LOSS_FN
from trinity.algorithm.entropy_loss_fn.entropy_loss_fn import EntropyLossFn
from trinity.algorithm.kl_fn import KL_FN
from trinity.algorithm.kl_fn.kl_fn import KLFn
from trinity.algorithm.policy_loss_fn import POLICY_LOSS_FN
from trinity.algorithm.policy_loss_fn.dpo_loss import DPOLossFn
from trinity.algorithm.policy_loss_fn.mix_policy_loss import MIXPolicyLossFn
from trinity.algorithm.policy_loss_fn.opmd_policy_loss import OPMDPolicyLossFn
from trinity.algorithm.policy_loss_fn.ppo_policy_loss import PPOPolicyLossFn
from trinity.algorithm.policy_loss_fn.sft_loss import SFTLossFn
from trinity.algorithm.sample_strategy import SAMPLE_STRATEGY
from trinity.algorithm.sample_strategy.mix_sample_strategy import MixSampleStrategy
from trinity.manager.config_registry.config_registry import CONFIG_GENERATORS
from trinity.manager.config_registry.model_config_manager import set_trainer_gpu_num
from trinity.utils.registry import Registry


@CONFIG_GENERATORS.register_config(
    default_value="grpo",
    other_configs={"mode": "both", "_current_default_config": GRPOAlgorithm.default_config()},
)
def set_algorithm_type(**kwargs):
    def on_change():
        if st.session_state["algorithm_type"] in ("dpo", "sft"):
            st.session_state["mode"] = "train"
        else:
            st.session_state["mode"] = "both"
        algorithm = ALGORITHM_TYPE.get(st.session_state["algorithm_type"])
        default_config = algorithm.default_config()
        st.session_state["_current_default_config"] = default_config
        for key, value in default_config.items():
            st.session_state[key] = value
        set_trainer_gpu_num()

    candidates = list(ALGORITHM_TYPE.modules.keys())
    st.selectbox(
        "Algorithm Type",
        options=candidates,
        format_func=lambda x: x.upper(),
        on_change=on_change,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=GRPOAlgorithm.default_config()["repeat_times"],
    visible=lambda: "repeat_times" in st.session_state["_current_default_config"],
    other_configs={
        "_grouped_adv_repeat_times": 2,
        "_not_grouped_adv_repeat_times": 1,
    },
)
def set_repeat_times(**kwargs):
    key = kwargs.get("key")
    grouped_adv_algorithms = [
        "grpo",
        "dapo",
        "opmd",
        "rloo",
    ]
    if st.session_state["algorithm_type"] in grouped_adv_algorithms:
        min_repeat_times = 2
        st.session_state[key] = st.session_state["_grouped_adv_repeat_times"]
    else:
        min_repeat_times = 1
        st.session_state[key] = st.session_state["_not_grouped_adv_repeat_times"]

    def on_change():
        if st.session_state["algorithm_type"] in grouped_adv_algorithms:
            st.session_state["_grouped_adv_repeat_times"] = st.session_state[key]
        else:
            st.session_state["_not_grouped_adv_repeat_times"] = st.session_state[key]

    st.number_input(
        "Repeat Times",
        min_value=min_repeat_times,
        help="`repeat_times` is used to set how many experiences each task can generate, "
        "and it must be greater than `1` when `algorithm_type` is `grpo`, `opmd` or 'rloo`.",
        on_change=on_change,
        **kwargs,
    )


# Sample_strategy Configs


@CONFIG_GENERATORS.register_config(
    default_value=GRPOAlgorithm.default_config()["sample_strategy"],
    visible=lambda: "sample_strategy" in st.session_state["_current_default_config"],
)
def set_sample_strategy(**kwargs):
    on_change = _create_on_change_callback("sample_strategy", SAMPLE_STRATEGY, **kwargs)
    candidates = list(SAMPLE_STRATEGY.modules.keys())
    st.selectbox(
        "Sample Strategy",
        candidates,
        help="The sample strategy used to obtain experiences.",
        on_change=on_change,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=MixSampleStrategy.default_args()["expert_data_ratio"],
    visible=lambda: st.session_state["sample_strategy"] == "mix",
)
def set_expert_data_ratio_in_sample_strategy(**kwargs):
    st.number_input(
        "Expert Data Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="The ratio of expert data to be used in the training.",
        **kwargs,
    )


# Advantage Configs


@CONFIG_GENERATORS.register_config(
    default_value=GRPOAlgorithm.default_config()["advantage_fn"],
    visible=lambda: "advantage_fn" in st.session_state["_current_default_config"],
)
def set_advantage_fn(**kwargs):
    on_change = _create_on_change_callback("advantage_fn", ADVANTAGE_FN, **kwargs)
    candidates = list(ADVANTAGE_FN.modules.keys())
    st.selectbox(
        "Advantage Function",
        options=candidates,
        format_func=lambda x: x.upper(),
        help="The advantage function used to compute advantages.",
        on_change=on_change,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=PPOAdvantageFn.default_args()["gamma"],
    visible=lambda: st.session_state["advantage_fn"] in {"ppo", "reinforceplusplus"},
)
def set_gamma_in_advantage_fn(**kwargs):
    st.number_input(r"Gamma :blue-badge[$\gamma$]", help="Discounted factor used in RL", **kwargs)


@CONFIG_GENERATORS.register_config(
    default_value=PPOAdvantageFn.default_args()["lam"],
    visible=lambda: st.session_state["advantage_fn"] == "ppo",
)
def set_lam_in_advantage_fn(**kwargs):
    st.number_input(
        r"Lambda :blue-badge[$\lambda$]",
        help="Lambda value when computing Generalized Advantage Estimation",
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=GRPOAdvantageFn.default_args()["epsilon"],
    visible=lambda: st.session_state["advantage_fn"] == "grpo",
)
def set_epsilon_in_advantage_fn(**kwargs):
    st.number_input(
        r"GRPO Epsilon",
        help=r"""
```python
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
```
""",
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=OPMDAdvantageFn.default_args()["opmd_baseline"],
    visible=lambda: st.session_state["advantage_fn"] == "opmd",
)
def set_opmd_baseline_in_advantage_fn(**kwargs):
    st.selectbox(
        "OPMD Baseline",
        ["mean", "logavgexp"],
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=OPMDAdvantageFn.default_args()["tau"],
    visible=lambda: st.session_state["advantage_fn"] == "opmd"
    and st.session_state["opmd_baseline_in_advantage_fn"] == "logavgexp",
)
def set_tau_in_advantage_fn(**kwargs):
    st.number_input("Tau for OPMD Adv.", min_value=0.0, format="%.1e", **kwargs)


# KL Loss Configs


@CONFIG_GENERATORS.register_config(
    default_value=GRPOAlgorithm.default_config()["kl_loss_fn"],
    visible=lambda: "kl_loss_fn" in st.session_state["_current_default_config"],
)
def set_kl_loss_fn(**kwargs):
    on_change = _create_on_change_callback("kl_loss_fn", KL_FN, **kwargs)
    candidates = list(KL_FN.modules.keys())
    st.selectbox(
        "KL Loss Type",
        options=candidates,
        format_func=lambda x: x.upper(),
        on_change=on_change,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=KLFn.default_args()["kl_coef"],
    visible=lambda: st.session_state["kl_loss_fn"] != "none",
)
def set_kl_coef_in_kl_loss_fn(**kwargs):
    st.number_input(
        r"KL Loss Coef :blue-badge[$\beta$]",
        min_value=0.0,
        max_value=1.0,
        format="%.1e",
        **kwargs,
    )


# KL Penalty Configs


@CONFIG_GENERATORS.register_config(
    default_value=GRPOAlgorithm.default_config()["kl_penalty_fn"],
    visible=lambda: "kl_penalty_fn" in st.session_state["_current_default_config"],
)
def set_kl_penalty_fn(**kwargs):
    on_change = _create_on_change_callback("kl_penalty_fn", KL_FN, **kwargs)
    candidates = list(KL_FN.modules.keys())
    st.selectbox(
        "KL Penalty Type",
        options=candidates,
        format_func=lambda x: x.upper(),
        on_change=on_change,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=KLFn.default_args()["adaptive"],
    visible=lambda: st.session_state["kl_penalty_fn"] != "none",
)
def set_adaptive_in_kl_penalty_fn(**kwargs):
    st.checkbox(
        "Adaptive KL Penalty",
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=KLFn.default_args()["kl_coef"],
    visible=lambda: st.session_state["kl_penalty_fn"] != "none",
)
def set_kl_coef_in_kl_penalty_fn(**kwargs):
    st.number_input(
        r"KL Penalty Coef",
        min_value=0.0,
        max_value=1.0,
        format="%.1e",
        **kwargs,
    )


# TODO: target_kl and horizon

# Policy Loss Configs


@CONFIG_GENERATORS.register_config(
    default_value=GRPOAlgorithm.default_config()["policy_loss_fn"],
    visible=lambda: "policy_loss_fn" in st.session_state["_current_default_config"],
)
def set_policy_loss_fn(**kwargs):
    on_change = _create_on_change_callback("policy_loss_fn", POLICY_LOSS_FN, **kwargs)
    candidates = list(POLICY_LOSS_FN.modules.keys())
    st.selectbox(
        "Policy Loss Fn",
        options=candidates,
        format_func=lambda x: x.upper(),
        on_change=on_change,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=PPOPolicyLossFn.default_args()["clip_range"],
    visible=lambda: st.session_state["policy_loss_fn"] in {"ppo", "mix"},
)
def set_clip_range_in_policy_loss_fn(**kwargs):
    st.number_input(
        "Clip Range",
        min_value=0.0,
        max_value=1.0,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=SFTLossFn.default_args()["loss_agg_mode"],
    visible=lambda: st.session_state["policy_loss_fn"] == "sft",
)
def set_sft_loss_agg_mode(**kwargs):
    candidates = [
        "token-mean",
        "seq-mean-token-sum",
        "seq-mean-token-mean",
        "seq-mean-token-sum-norm",
    ]
    st.selectbox(
        "SFT Loss Aggregation Mode",
        candidates,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=DPOLossFn.default_args()["beta"],
    visible=lambda: st.session_state["policy_loss_fn"] == "dpo",
)
def set_beta_in_policy_loss_fn(**kwargs):
    st.number_input(
        "Beta for DPO",
        min_value=0.0,
        max_value=1.0,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=DPOLossFn.default_args()["label_smoothing"],
    visible=lambda: st.session_state["policy_loss_fn"] == "dpo",
)
def set_label_smoothing_in_policy_loss_fn(**kwargs):
    st.number_input(
        "Label Smoothing",
        min_value=0.0,
        max_value=1.0,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=OPMDPolicyLossFn.default_args()["tau"],
    visible=lambda: st.session_state["policy_loss_fn"] == "opmd",
)
def set_tau_in_policy_loss_fn(**kwargs):
    st.number_input("Tau for OPMD Loss", min_value=0.0, format="%.1e", **kwargs)


@CONFIG_GENERATORS.register_config(
    default_value=MIXPolicyLossFn.default_args()["mu"],
    visible=lambda: st.session_state["policy_loss_fn"] == "mix",
)
def set_mu_in_policy_loss_fn(**kwargs):
    st.number_input("Mu for Mix Policy Loss", min_value=0.0, **kwargs)


# Entropy Loss Configs


@CONFIG_GENERATORS.register_config(
    default_value=GRPOAlgorithm.default_config()["entropy_loss_fn"],
    visible=lambda: "entropy_loss_fn" in st.session_state["_current_default_config"],
)
def set_entropy_loss_fn(**kwargs):
    on_change = _create_on_change_callback("entropy_loss_fn", ENTROPY_LOSS_FN, **kwargs)
    candidates = list(ENTROPY_LOSS_FN.modules.keys())
    st.selectbox(
        "Entropy Loss Function",
        options=candidates,
        on_change=on_change,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=EntropyLossFn.default_args()["entropy_coef"],
    visible=lambda: st.session_state["entropy_loss_fn"] != "none",
)
def set_entropy_coef_in_entropy_loss_fn(**kwargs):
    st.number_input(
        "Entropy Coeff",
        min_value=0.0,
        max_value=1.0,
        format="%.1e",
        **kwargs,
    )


# define on_change
def _create_on_change_callback(key_name: str, registry: Registry, **kwargs):
    """Creates an on_change callback to update dependent configs."""

    def on_change():
        value = st.session_state[kwargs.get("key", key_name)]
        value_class = registry.get(value)
        if value_class:
            default_args = value_class.default_args()
            for arg_key, arg_value in default_args.items():
                full_key = f"{arg_key}_in_{key_name}"
                st.session_state[full_key] = arg_value

    return on_change
