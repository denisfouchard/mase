# %%
import logging
from ultralytics import YOLO
from chop import MaseGraph
import torch.fx as fx
import torch.nn as nn
import torch
import ultralytics

import sys
import ipdb

sys.breakpointhook = ipdb.set_trace

# Load a pretrained YOLO model
model = YOLO("yolov8n-cls.pt", task="classification", verbose=True)  # Choose the appropriate model

# Define a safe wrapper for torch.cat to avoid tracing its internals
@fx.wrap
def safe_cat(x, dim):
    return torch.cat(tuple(x), dim=dim)


def safe_settatr(obj, name, value):
    if isinstance(value, int):
        setattr(obj, name, value)
    elif isinstance(value, list):
        setattr(obj, name, value)
    elif isinstance(value, str):
        setattr(obj, name, value)
    elif isinstance(value, float):
        setattr(obj, name, value)


# FX-safe wrapper for Concat
class FXSafeConcat(nn.Module):
    def __init__(self, orig_module):
        super().__init__()
        attrs = vars(orig_module)
        for name, value in attrs.items():
            safe_settatr(self, name, value)
        self.d = orig_module.d

    def forward(self, x):
        return safe_cat(x, dim=self.d)


# Define a safe wrapper for Detect module calls
@fx.wrap
def safe_detect(
    module,
    *args,
):
    return module(
        *args,
    )


# FX-safe wrapper for Detect
class FXSafeDetect(nn.Module):
    def __init__(self, orig_module):
        super().__init__()
        self.orig_module = orig_module
        # Instantiate all the original module's parameters
        attrs = vars(orig_module)
        for name, value in attrs.items():
            safe_settatr(self, name, value)

    def forward(self, *args):
        return (safe_detect(self.orig_module, *args),)


# Define a safe wrapper for C2f module calls
@fx.wrap
def safe_c2f(
    module,
    x,
):
    return module(x)


# FX-safe wrapper for C2f
class FXSafeC2f(nn.Module):
    def __init__(self, orig_module):
        super().__init__()

        # Instantiate all the original module's parameters
        attrs = vars(orig_module)
        for name, value in attrs.items():
            safe_settatr(self, name, value)

        self.orig_module = orig_module

    def forward(self, x, **kwargs):
        return safe_c2f(self.orig_module, x)


# Replace problematic modules in the model with FX-safe versions
# for name, module in model.model.model.named_children():
#     if isinstance(module, ultralytics.nn.modules.conv.Concat):
#         print(f"Replacing module {name} with FXSafeConcat")
#         setattr(model.model.model, name, FXSafeConcat(module))
#     elif isinstance(module, ultralytics.nn.modules.head.Detect):
#         print(f"Replacing module {name} with FXSafeDetect")
#         setattr(model.model.model, name, FXSafeDetect(module))
#     elif isinstance(module, ultralytics.nn.modules.block.C2f):
#         print(f"Replacing module {name} with FXSafeC2f")
#         setattr(model.model.model, name, FXSafeC2f(module))


cf_args = {
    # "x": torch.randn(1, 3, 640, 640),
    # "*args": [],
    "profile": False,
    "visualize": False,
    "augment": False,
    "embed": None,
}

# model.model.patched_nodes = {
#     "functions": [
#         torch.cat,
#     ],
#     "modules": [
#         ultralytics.nn.modules.head.Detect,
#         ultralytics.nn.modules.C2f,
#     ],
#     "layers": [],
# }

# %%
# breakpoint()
mg = MaseGraph(model.model, cf_args=cf_args)
# Set custom_ops
CUSTOM_OPS = {
    "modules": {FXSafeC2f: "", FXSafeConcat: "", FXSafeDetect: ""},
    # "functions": {safe_cat: "", safe_detect: "", safe_c2f: ""},
}
setattr(mg.model, "custom_ops", CUSTOM_OPS)

print("MaseGraph successfully created:")
print(mg)

mg.model.patched_op_names = [
    "safe_cat",
    "safe_detect",
    "safe_c2f",
    "safe_settatr",
    "safe_list_create",
    "safe_append",
    "safe_unbind",
]

# %%

import chop.passes as passes
from chop.passes.graph.analysis.add_metadata.common_metadata_layers import func_data

# Add metadata for the custom functions
func_data["safe_c2f"] = {
    "module": "C2f",
    "input": "data_in",
    "c1": "config",
    "c2": "config",
    "shortcut": False,
    "g": "config",
    "e": "config",
}
func_data["safe_detect"] = {"module": "detect", "input": "data_in"}
func_data["safe_cat"] = {"module": "concat", "input": "data_in", "dim": "config"}
func_data["safe_settatr"] = {
    "module": "settatr",
    "input": "data_in",
    "name": "config",
    "value": "config",
}
func_data["safe_list_create"] = {
    "module": "list_create",
    "m": "config",
    "x": "data_in",
    "y": "config",
}
func_data["safe_append"] = {
    "module": "append",
    "x": "data_in",
}
func_data["safe_unbind"] = {
    "module": "unbind",
    "x": "data_in",
    "dim": "config",
}

param = next(mg.model.model.parameters())[1]

dummy_input = torch.rand(1, 3, 640, 640, dtype=param.dtype).to(param.device)

breakpoint()

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(
    mg,
    {
        "dummy_in": {
            "x": dummy_input,
            "profile_1": False,
            "visualize_1": False,
            "augment_1": False,
            "embed_1": None,
        },
        "add_value": True,
    },
)
mg, _ = passes.add_software_metadata_analysis_pass(mg, None)

# %% Quantization

import chop.passes as passes

quantization_config = {
    "by": "type",
    "default": {
        "config": {
            "name": None,
        }
    },
    "linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 16,
            "data_in_frac_width": 8,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
    "conv2d": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 6,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

from chop.passes.graph.utils import deepcopy_mase_graph

orig_mg = deepcopy_mase_graph(mg)

mg, _ = passes.quantize_transform_pass(
    mg,
    pass_args=quantization_config,
)
mg, _ = passes.summarize_quantization_analysis_pass(
    mg,
    pass_args={"save_dir": "quantize_summary", "original_graph": orig_mg},
)

# %%

torch.save(mg.model.state_dict(prefix="model."), "classifier.pt")
print(model.state_dict().keys())
breakpoint()
metrics = model.val(data="imagenet10")
print(metrics)
