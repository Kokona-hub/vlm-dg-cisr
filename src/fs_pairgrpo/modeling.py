from __future__ import annotations

from typing import Any


def require_huggingface_training_backend(config: dict[str, Any]) -> None:
    backend = str(config.get("backend", "huggingface")).lower()
    name_or_path = str(config.get("name_or_path", ""))
    if backend != "huggingface":
        raise ValueError(
            "QLoRA-SFT and GRPO require model.backend='huggingface'. Ollama exposes inference "
            "only and cannot provide trainable weights or gradients."
        )
    if name_or_path.startswith("ollama:") or name_or_path == "qwen3-vl:8b":
        raise ValueError(
            "The Ollama model name cannot be used for training. Set model.name_or_path to a "
            "Hugging Face checkpoint directory containing config.json and *.safetensors."
        )


def dtype_from_name(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(mapping)}")
    return mapping[name]


def quantization_config(config: dict[str, Any]):
    if not bool(config.get("load_in_4bit", True)):
        return None
    from transformers import BitsAndBytesConfig

    dtype = dtype_from_name(str(config.get("dtype", "bfloat16")))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
    )


def lora_config(config: dict[str, Any]):
    from peft import LoraConfig

    return LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config.get("r", 16)),
        lora_alpha=int(config.get("alpha", 32)),
        lora_dropout=float(config.get("dropout", 0.05)),
        bias="none",
        target_modules=config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )


def processor_init_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", True))
    }
    for name in ("min_pixels", "max_pixels"):
        if model_config.get(name) is not None:
            kwargs[name] = int(model_config[name])
    return kwargs


def model_init_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    import torch

    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
        "dtype": dtype_from_name(str(model_config.get("dtype", "bfloat16"))),
        "attn_implementation": str(model_config.get("attn_implementation", "sdpa")),
    }
    quantization = quantization_config(model_config)
    if quantization is not None:
        from accelerate import PartialState

        kwargs["quantization_config"] = quantization
        kwargs["device_map"] = {"": PartialState().local_process_index}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    return kwargs


def prepare_kbit_model(model, model_config: dict[str, Any]):
    """Prepare QLoRA training while avoiding reentrant-checkpoint input warnings."""
    from peft import prepare_model_for_kbit_training

    enabled = bool(model_config.get("gradient_checkpointing", True))
    try:
        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=enabled,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        # Compatibility fallback for older PEFT installations.
        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=enabled,
        )


def load_vlm_with_adapter(
    base_model_path: str,
    adapter_path: str,
    model_config: dict[str, Any],
    trainable: bool = True,
):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        **model_init_kwargs(model_config),
    )
    if bool(model_config.get("load_in_4bit", True)):
        model = prepare_kbit_model(model, model_config)
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)


def load_vlm(base_model_path: str, model_config: dict[str, Any]):
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        base_model_path, **model_init_kwargs(model_config)
    )


def load_vlm_with_trainable_and_reference_adapter(
    base_model_path: str,
    adapter_path: str,
    model_config: dict[str, Any],
):
    """Load two adapter copies: one trainable policy and one frozen SFT reference."""
    from peft import PeftModel

    model = load_vlm(base_model_path, model_config)
    if bool(model_config.get("load_in_4bit", True)):
        model = prepare_kbit_model(model, model_config)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.load_adapter(adapter_path, adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
    return model


def set_active_adapters(model, adapter_names: str | list[str]) -> None:
    """Activate one adapter or an additive LoRA composition.

    CRLoRA needs the prediction policy ``Shared + Domain_d`` during source
    updates and ``Shared`` alone for pseudo-target and real-target inference.
    Recent PEFT releases support a list here.  Failing loudly is preferable to
    silently training an unintended single-adapter policy on older releases.
    """
    names = [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
    if len(names) == 1:
        model.set_adapter(names[0])
        return

    # Some PEFT releases support lists in LoraLayer.set_adapter but not in
    # PeftModel.set_adapter. Set the model to a valid single adapter first,
    # then update each LoRA layer directly. This keeps additive inference
    # compatible with the server's installed PEFT implementation.
    model.set_adapter(names[0])
    layer_count = 0
    for module in model.modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "set_adapter"):
            continue
        try:
            module.set_adapter(names)
        except (TypeError, ValueError, NotImplementedError):
            # Older LoraLayer implementations expose the active adapter as a
            # plain attribute even when their setter only accepts one name.
            module.active_adapter = names
        layer_count += 1
    if layer_count == 0:
        raise RuntimeError(
            "No PEFT LoRA layers were found while enabling additive adapters."
        )


def load_vlm_with_consensus_residual_adapters(
    base_model_path: str,
    shared_adapter_path: str,
    model_config: dict[str, Any],
    lora_config_value,
    source_domains: list[str],
    shared_adapter_name: str = "shared",
    reference_adapter_name: str = "reference",
    domain_adapter_prefix: str = "domain_",
):
    """Load one Shared LoRA, a frozen reference, and zero-initialized Domain LoRAs.

    Every adapter uses the same nominal LoRA configuration.  This is required
    because consensus and residual gradients are routed in a common parameter
    coordinate system; ranks or target modules must therefore not differ.
    """
    import torch
    from peft import PeftModel

    if len(set(source_domains)) != len(source_domains):
        raise ValueError("Source domains must be unique when creating Domain LoRAs.")
    model = load_vlm(base_model_path, model_config)
    if bool(model_config.get("load_in_4bit", True)):
        model = prepare_kbit_model(model, model_config)
    model = PeftModel.from_pretrained(
        model,
        shared_adapter_path,
        adapter_name=shared_adapter_name,
        is_trainable=True,
    )
    model.load_adapter(
        shared_adapter_path,
        adapter_name=reference_adapter_name,
        is_trainable=False,
    )
    for domain in source_domains:
        adapter_name = f"{domain_adapter_prefix}{domain}"
        model.add_adapter(adapter_name, lora_config_value)
    # Give every Domain LoRA the Shared adapter's learned A basis while keeping
    # its B factor zero.  Thus each domain branch starts as an exact no-op, and
    # initial gradient differences reflect source data rather than random bases.
    with torch.no_grad():
        named_parameters = dict(model.named_parameters())
        for domain in source_domains:
            adapter_name = f"{domain_adapter_prefix}{domain}"
            for name, parameter in named_parameters.items():
                marker = f".{adapter_name}."
                if marker not in name:
                    continue
                shared_name = name.replace(marker, f".{shared_adapter_name}.", 1)
                if ".lora_A." in name and shared_name in named_parameters:
                    parameter.copy_(named_parameters[shared_name])
                elif ".lora_B." in name:
                    parameter.zero_()
    set_active_adapters(model, shared_adapter_name)
    return model
