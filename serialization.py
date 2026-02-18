from dataclasses import asdict
from typing import Protocol, Any, Literal
import json
import importlib


class Dictable(Protocol):
    def asdict(self) -> dict:
        ...


def dumps(obj: Dictable, format: Literal["json", "msgpack"] = "json", minify: bool = True) -> str:
    """Serialize any Dictable object to JSON or msgpack."""
    if format != "json":
        raise NotImplementedError
    dict_version = obj.asdict()
    module = obj.__class__.__module__
    if module.strip():
        module = module + "."
    else:
        module = ""
    dict_version['__class__'] = f"{module}{obj.__class__.__name__}"
    kwargs = {}
    if minify:
        kwargs["separators"] = (',', ':')
    return json.dumps(dict_version, **kwargs)


def load(json_str: str, cls: type | None = None) -> Any:
    """Deserialize JSON string to a class instance."""
    data = json.loads(json_str)
    
    # If explicit class is provided, use it
    if cls is not None:
        if "__class__" in data:
            data.pop('__class__')  # Remove class info if present
        return cls(**data)
    
    # Otherwise, extract class name from JSON data
    cls_name = data.pop('__class__', None)
    if cls_name is None:
        # no class info, just return whatever json.loads gave us, list or dict
        return data
    
    # Handle the case where class name includes module path
    if '.' in cls_name:
        # Split into module and class name
        module_name, class_name = cls_name.rsplit('.', 1)
        try:
            # Import the module
            module = importlib.import_module(module_name)
            # Get the class from the module
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not import class '{cls_name}': {e}")
    else:
        # Handle built-in classes or classes in current namespace
        try:
            # Try to find it in built-ins first
            cls = globals().get(cls_name) or locals().get(cls_name)
            if cls is None:
                # Try importing from builtins
                import builtins
                cls = getattr(builtins, cls_name, None)
            if cls is None:
                raise ValueError(f"Class '{cls_name}' not found")
        except Exception as e:
            raise ValueError(f"Could not find class '{cls_name}': {e}")
    
    return cls(**data)
