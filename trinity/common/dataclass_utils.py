from dataclasses import fields, is_dataclass
from typing import Any, Union, get_args, get_origin

from omegaconf import OmegaConf


def build_dataclass_from_mapping(dataclass_type: type[Any], data: Any) -> Any:
    """
    Convert a mapping (dict, OmegaConf, etc.) to a dataclass instance, recursively.
    """
    if data is None:
        return None
    if is_dataclass(data):
        return data
    mapping = _normalize_mapping(data)
    kwargs = {}
    for dataclass_field in fields(dataclass_type):
        if dataclass_field.name in mapping:
            kwargs[dataclass_field.name] = _convert_dataclass_field(
                dataclass_field.type, mapping[dataclass_field.name]
            )
    return dataclass_type(**kwargs)


def _normalize_mapping(raw_data: Any) -> dict[str, Any]:
    if isinstance(raw_data, dict):
        return raw_data
    return OmegaConf.to_container(OmegaConf.create(raw_data), resolve=True)  # type: ignore[return-value]


def _strip_optional(value_type: Any) -> Any:
    origin = get_origin(value_type)
    if origin is not Union:
        return value_type
    non_none_types = [arg for arg in get_args(value_type) if arg is not type(None)]
    return non_none_types[0] if len(non_none_types) == 1 else value_type


def _convert_dataclass_field(expected_type: Any, value: Any) -> Any:
    if value is None:
        return None
    expected_type = _strip_optional(expected_type)
    origin = get_origin(expected_type)

    if isinstance(expected_type, type) and is_dataclass(expected_type):
        if is_dataclass(value) or not isinstance(value, dict):
            return value
        return build_dataclass_from_mapping(expected_type, value)

    if origin in (list, tuple, set):
        item_type = get_args(expected_type)[0] if get_args(expected_type) else Any
        converted_items = [_convert_dataclass_field(item_type, item) for item in value]
        if origin is tuple:
            return tuple(converted_items)
        if origin is set:
            return set(converted_items)
        return converted_items

    if origin is dict:
        args = get_args(expected_type)
        value_type = args[1] if len(args) > 1 else Any
        return {k: _convert_dataclass_field(value_type, v) for k, v in value.items()}

    return value
