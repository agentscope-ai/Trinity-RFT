import os
from typing import Optional

from trinity.utils.log import get_logger


class Converter:
    def __init__(self, base_model_dir: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.base_model_dir = base_model_dir
        self.base_model = None
        self._init_process_group = False
        self.checkpoint_converter = None

    def init_base_model(self) -> bool:
        if not self.base_model_dir:
            self.logger.error(
                "Base model directory is not specified. "
                "Please specify it with `--base-model-dir /path/to/model`."
            )
            return False
        if self.base_model is not None:
            return True
        try:
            self.base_model, _ = self._get_config_and_empty_model(self.base_model_dir)
        except Exception:
            self.logger.error(
                f"Failed to initialize base model from {self.base_model_dir}", exc_info=True
            )
            return False
        return True

    def init_process_group(self):
        if self._init_process_group:
            return

        import torch
        from verl.utils.device import get_nccl_backend
        from verl.utils.distributed import set_numa_affinity

        if "WORLD_SIZE" not in os.environ:
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

        set_numa_affinity()
        torch.distributed.init_process_group(get_nccl_backend())
        self._init_process_group = True

    def init_checkpoint_converter(self, checkpoint_dir) -> bool:
        if self.checkpoint_converter is not None:
            return True
        if not os.path.basename(checkpoint_dir).startswith("global_step_"):
            self.logger.error(f"Invalid checkpoint directory {checkpoint_dir}.")
            return False

        actor_ckpt_dir = os.path.join(checkpoint_dir, "actor")
        huggingface_dir = os.path.join(actor_ckpt_dir, "huggingface")
        if not os.path.exists(os.path.join(huggingface_dir, "config.json")):
            if not self.init_base_model():
                self.logger.error(
                    f"Failed to load base model from {self.base_model_dir}, "
                    "please check if the model exists."
                )
                return False
            self.base_model.config.save_pretrained(huggingface_dir)

        from trinity.common.models.utils import get_megatron_converter

        self.init_process_group()
        self.checkpoint_converter = get_megatron_converter(actor_ckpt_dir)
        return True

    def _get_config_and_empty_model(self, model_dir: str):
        import torch
        import transformers
        from accelerate import init_empty_weights

        model_config = transformers.AutoConfig.from_pretrained(model_dir)

        if "ForTokenClassification" in model_config.architectures[0]:
            from transformers import AutoModelForTokenClassification

            auto_model_cls = AutoModelForTokenClassification
        elif "ForCausalLM" in model_config.architectures[0]:
            from transformers import AutoModelForCausalLM

            auto_model_cls = AutoModelForCausalLM
        elif "ForConditionalGeneration" in model_config.architectures[0]:
            # Handle different transformers versions for Vision2Seq models
            import transformers
            from packaging import version

            if version.parse(transformers.__version__) >= version.parse("4.54.0"):
                # transformers >= 4.54.0 uses AutoModelForImageTextToText
                from transformers import AutoModelForImageTextToText

                auto_model_cls = AutoModelForImageTextToText
            else:
                # transformers < 4.54.0 uses AutoModelForVision2Seq
                from transformers import AutoModelForVision2Seq

                auto_model_cls = AutoModelForVision2Seq
        else:
            raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

        with init_empty_weights():
            model = auto_model_cls.from_config(model_config, dtype=torch.bfloat16)
        model.to_empty(device="cpu")

        return model, auto_model_cls

    def convert(self, checkpoint_dir: str) -> None:
        if os.path.basename(checkpoint_dir).startswith("global_step_"):
            import torch

            actor_ckpt_dir = os.path.join(checkpoint_dir, "actor")
            huggingface_dir = os.path.join(actor_ckpt_dir, "huggingface")
            model = None
            if os.path.exists(huggingface_dir):
                has_hf_checkpoint = True
                try:
                    model, auto_model_cls = self._get_config_and_empty_model(huggingface_dir)
                    auto_model_cls.from_pretrained(huggingface_dir)
                except Exception:
                    self.logger.debug(
                        f"Incomplete or invalid Hugging Face checkpoint in {huggingface_dir}, will re-convert.",
                        exc_info=True,
                    )
                    has_hf_checkpoint = False

                if has_hf_checkpoint:
                    return
            if model is None:
                if not self.init_base_model():
                    self.logger.error(
                        f"Failed to load base model from {self.base_model_dir}, please check if the model exists."
                    )
                    return
                model = self.base_model

            self.logger.info(f"Converting {checkpoint_dir} to huggingface format...")
            dist_cpkt_dir = os.path.join(actor_ckpt_dir, "dist_ckpt")
            try:
                if os.path.exists(dist_cpkt_dir):  # megatron
                    if not self.init_checkpoint_converter(checkpoint_dir):
                        return
                    state_dict = self.checkpoint_converter.get_state_dict(actor_ckpt_dir)
                else:  # fsdp
                    from trinity.common.models.utils import (
                        load_fsdp_state_dict_from_verl_checkpoint,
                    )

                    state_dict = load_fsdp_state_dict_from_verl_checkpoint(actor_ckpt_dir)
            except Exception:
                self.logger.error(
                    f"Failed to convert {checkpoint_dir} to huggingface format.",
                    exc_info=True,
                )
                return

            state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            model.save_pretrained(huggingface_dir, state_dict=state_dict)
            self.logger.info(f"Saved huggingface checkpoint to {huggingface_dir}")

        else:  # recursive search
            for sub_dir in os.listdir(checkpoint_dir):
                sub_dir_path = os.path.join(checkpoint_dir, sub_dir)
                if os.path.isdir(sub_dir_path):
                    self.convert(sub_dir_path)
