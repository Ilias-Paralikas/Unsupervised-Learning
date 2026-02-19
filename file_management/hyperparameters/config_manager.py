import yaml
import os
from .registry import REGISTRY, get_component

def to_serializable(item):
    """Recursively converts objects to their string names for YAML."""
    if isinstance(item, dict):
        return {k: to_serializable(v) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        return [to_serializable(i) for i in item]
    elif isinstance(item, type):  # It's a class
        return item.__name__
    elif callable(item): # It's a function/lambda
        # Reverse lookup in registry to find the name
        for name, obj in REGISTRY.items():
            if obj == item:
                return name
        return item.__name__ if hasattr(item, '__name__') else str(item)
    return item


class CustomDumper(yaml.SafeDumper):
    """Custom dumper to force flow style (brackets) only for lists."""
    def represent_sequence(self, tag, sequence, flow_style=None):
        # Always use flow_style (horizontal) for lists/tuples
        return super().represent_sequence(tag, sequence, flow_style=True)

def save_config(config, path):
    serializable_config = to_serializable(config)
    
    with open(path, 'w') as f:
        yaml.dump(
            serializable_config, 
            f, 
            Dumper=CustomDumper, 
            sort_keys=False, 
            width=1000
        )
    print(f"Config saved to {path}")
def load_config(path):
    """Loads YAML and 're-hydrates' the PyTorch objects."""
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    def rehydrate(item):
        if isinstance(item, dict):
            return {k: rehydrate(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [rehydrate(i) for i in item]
        elif isinstance(item, str) and item in REGISTRY:
            return REGISTRY[item]
        return item

    return rehydrate(raw_config)