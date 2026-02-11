"""Monkey patching for 'kimi_vl' models."""


def kimi_vl_monkey_patch_decorator(func):
    """
    A decorator that applies temporary monkey patches for 'kimi_vl' models before
    the decorated function runs, and restores the original state afterward.

    The patch is applied only if:
      - The model's config.json exists and specifies "model_type": "kimi_vl"
      - The installed transformers version is >= 4.51.0

    Patches include:
      1. Replacing `transformers.activations.PytorchGELUTanh` with `GELUTanh`
      2. Wrapping `importlib.util.spec_from_file_location` to inject DeepseekV3 classes

    The decorator automatically extracts `model_path` and `override_model_config`
    from the function's arguments using `inspect.signature`, regardless of whether
    they are passed as positional or keyword arguments.
    """
    import importlib
    import inspect
    import json
    import os
    from functools import wraps

    import transformers
    from packaging import version

    transformers_version = transformers.__version__
    sig = inspect.signature(func)  # Analyze function signature once at decoration time

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind actual arguments to parameter names (handles pos/kw/defaults)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Extract required parameters safely by name
        if "model_path" in bound_args.arguments:  # actor/ref worker
            model_path = bound_args.arguments["model_path"]
        elif "model_config" in bound_args.arguments:  # verl config check
            model_path = bound_args.arguments["model_config"].path
        elif "self" in bound_args.arguments:  # critic worker
            model_path = bound_args.arguments["self"].config.model.path

        # Track patch state for cleanup
        kimi_vl_patch_applied = False
        origin_spec_from_file_location = None
        origin_PytorchGELUTanh = None

        try:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    json_hf_config = json.load(f)

                # Check if model requires special patching
                if json_hf_config.get("model_type") == "kimi_vl" and version.parse(
                    transformers_version
                ) >= version.parse("4.51.0"):
                    # Save original values for restoration
                    origin_PytorchGELUTanh = getattr(
                        transformers.activations, "PytorchGELUTanh", None
                    )
                    origin_spec_from_file_location = importlib.util.spec_from_file_location

                    # Patch 1: Replace PytorchGELUTanh
                    transformers.activations.PytorchGELUTanh = transformers.activations.GELUTanh

                    # Patch 2: Wrap spec_from_file_location to inject DeepseekV3 classes
                    def patched_spec_from_file_location(*args_spec, **kwargs_spec):
                        spec = origin_spec_from_file_location(*args_spec, **kwargs_spec)
                        if spec and hasattr(spec, "loader") and spec.loader:
                            original_exec_module = spec.loader.exec_module

                            def patched_exec_module(module):
                                original_exec_module(module)
                                # Inject DeepseekV3* classes from transformers into the module
                                for attr_name in dir(module):
                                    if attr_name.startswith("DeepseekV3") and hasattr(
                                        transformers, attr_name
                                    ):
                                        setattr(module, attr_name, getattr(transformers, attr_name))
                                    elif attr_name in {
                                        "KimiVLPreTrainedModel",
                                        "KimiVLForConditionalGeneration",
                                    }:
                                        setattr(
                                            getattr(module, attr_name),
                                            "supports_gradient_checkpointing",
                                            True,
                                        )
                                        setattr(getattr(module, attr_name), "_supports_sdpa", True)

                            spec.loader.exec_module = patched_exec_module
                        return spec

                    importlib.util.spec_from_file_location = patched_spec_from_file_location

                    kimi_vl_patch_applied = True

            # Call the original function
            return func(*args, **kwargs)

        finally:
            # Always restore original state, even if an exception occurred
            if kimi_vl_patch_applied:
                # Restore PytorchGELUTanh
                if origin_PytorchGELUTanh is not None:
                    transformers.activations.PytorchGELUTanh = origin_PytorchGELUTanh
                else:
                    # Remove attribute if it didn't exist originally
                    if hasattr(transformers.activations, "PytorchGELUTanh"):
                        delattr(transformers.activations, "PytorchGELUTanh")

                # Restore spec_from_file_location
                if origin_spec_from_file_location is not None:
                    importlib.util.spec_from_file_location = origin_spec_from_file_location

    return wrapper
