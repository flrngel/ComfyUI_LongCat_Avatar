"""Layer streaming wrapper for memory-efficient inference.
Keeps most transformer/decoder layers on CPU pinned memory and streams them
to GPU on demand, using a secondary CUDA stream to prefetch upcoming layers
so that data transfer overlaps with compute.
General-purpose: works with any ``nn.Module`` whose forward iterates over a
``nn.ModuleList`` attribute (e.g. ``transformer_blocks``, ``layers``).
Each layer is evicted back to CPU immediately after its forward completes,
and prefetch uses modular indexing so the last layer's prefetch wraps around
to prepare early layers for the next forward pass.
Example
-------
>>> model = build_my_model(device=torch.device("cpu"))
>>> model = LayerStreamingWrapper(
...     model,
...     layers_attr="transformer_blocks",
...     target_device=torch.device("cuda:0"),
...     prefetch_count=2,
... )
>>> out = model(inputs)            # hooks handle layer streaming
>>> model.teardown()               # move everything back to CPU
"""

from __future__ import annotations

import functools
import itertools
import logging
from typing import Any,TypeVar
import gc
import torch
from torch import nn
from contextlib import contextmanager
from collections.abc import Iterator

logger = logging.getLogger(__name__)
_M = TypeVar("_M", bound=torch.nn.Module)
T = TypeVar("T")


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# LayerStreamingWrapper from https://github.com/Lightricks/LTX-2

class SimpleLayerStreamingWrapper_Dual(nn.Module):
    """Simplified layer streaming wrapper with support for multi-module offloading."""
    
    def __init__(
        self,
        model: nn.Module,
        layers_attrs: list[str],  # changed to a list to support multiple module paths
        target_device: torch.device,
        active_count: int = 1,
    ) -> None:
        super().__init__()
        self._model = model
        self._layers_attrs = layers_attrs
        self._target_device = target_device
        self._active_count = active_count
        
        # Resolve and store all modules that need to be offloaded
        self._layer_groups: list[nn.ModuleList] = []
        self._stores: list[_SimpleLayerStore] = []
        
        for attr in self._layers_attrs:
            layers = _resolve_attr(model, attr)
            self._layer_groups.append(layers)
            self._stores.append(_SimpleLayerStore(layers, self._target_device))
        
        # Move non-layer parameters to GPU
        self._move_non_layer_params_to_gpu()
        
        # Register hooks for all module groups
        self._register_simple_hooks()
    
    def _move_non_layer_params_to_gpu(self) -> None:
        """Move non-layer parameters to GPU, excluding all module parameters subject to streaming offload."""
        layer_tensor_ids = set()
        # Collect parameter IDs for all modules to be offloaded
        for layers in self._layer_groups:
            for layer in layers:
                for t in itertools.chain(layer.parameters(), layer.buffers()):
                    layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying model."""
        try:
            # First try to get the attribute from the wrapper itself
            return super().__getattr__(name)
        except AttributeError:
            # If that fails, get it from the underlying model
            return getattr(self._model, name)
    
    def _register_simple_hooks(self) -> None:
        """Register simple load/unload hooks for all module groups."""
        # Iterate over each module group and its corresponding store
        for layers, store in zip(self._layer_groups, self._stores):
            idx_map = {id(layer): idx for idx, layer in enumerate(layers)}
            
            def _pre_hook(module: nn.Module, input, *, idx: int, s: _SimpleLayerStore):
                # Load the current layer to GPU
                s.load_layer_to_gpu(idx, module)
                # Record the stream to prevent memory from being reclaimed prematurely
                for param in itertools.chain(module.parameters(), module.buffers()):
                    param.data.record_stream(torch.cuda.current_stream(self._target_device))
            
            def _post_hook(module: nn.Module, input, output, *, idx: int, s: _SimpleLayerStore):
                # Immediately move the layer back to CPU after processing
                s.unload_layer_from_gpu(idx, module)
            
            for layer in layers:
                idx = idx_map[id(layer)]
                # Use functools.partial to pass the corresponding store instance to the hook
                pre_hook = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx, s=store))
                post_hook = layer.register_forward_hook(functools.partial(_post_hook, idx=idx, s=store))

@contextmanager
def _streaming_model(
    model: _M,
    layers_attr,  # accepts str or list[str]
    target_device: torch.device,
    prefetch_count: int,
) -> Iterator[_M]:
    """Wrap *model* with :class:`LayerStreamingWrapper`, yield it, then tear down."""
    # Automatically route to the appropriate wrapper based on the type of layers_attr
    if isinstance(layers_attr, list):
        wrapped = SimpleLayerStreamingWrapper_Dual(
            model,
            layers_attrs=layers_attr,
            target_device=target_device,
            active_count=prefetch_count,
        )
    else:
        wrapped = SimpleLayerStreamingWrapper(
            model,
            layers_attr=layers_attr,
            target_device=target_device,
            active_count=prefetch_count,
        )
        
    try:
        yield wrapped  # type: ignore[misc]
    finally:
        wrapped.to("cpu")
        cleanup_memory()
        torch.cuda.synchronize(device=target_device)
        try:
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
        except Exception:
            print("Host empty cache cleanup failed; ignoring.", exc_info=True)



@contextmanager
def _streaming_model_(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
    prefetch_count: int,
) -> Iterator[_M]:
    """Wrap *model* with :class:`LayerStreamingWrapper`, yield it, then tear down."""
    wrapped = SimpleLayerStreamingWrapper(
        model,
        layers_attr=layers_attr,
        target_device=target_device,
        active_count=prefetch_count,
    )
    try:
        yield wrapped  # type: ignore[misc]
    finally:
        wrapped.to("cpu")
        cleanup_memory()
        # Flush the host (pinned) memory cache so that freed pinned pages are
        # returned to the OS.  Without this, sequential streaming models
        # (e.g. text encoder then transformer) exhaust host memory because the
        # CachingHostAllocator keeps freed blocks cached indefinitely.
        torch.cuda.synchronize(device=target_device)
        try:
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
        except Exception:
            print("Host empty cache cleanup failed; ignoring.", exc_info=True)


def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj

# edit from LayerStreamingWrapper from https://github.com/Lightricks/LTX-2

class _SimpleLayerStore:
    """Simplified layer store with on-demand loading and immediate release."""
    
    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)
        
        # Keep a reference to the original parameters on the CPU side
        self._cpu_params: list[dict[str, torch.Tensor]] = []
        for layer in layers:
            cpu_copy = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                cpu_copy[name] = tensor.data.cpu()  # kept on CPU
            self._cpu_params.append(cpu_copy)
    
    def load_layer_to_gpu(self, idx: int, layer: nn.Module) -> None:
        """Load the specified layer to GPU."""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name].to(self.target_device)
    
    def unload_layer_from_gpu(self, idx: int, layer: nn.Module) -> None:
        """Unload the specified layer from GPU back to CPU."""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name]  # restore to CPU copy


class SimpleLayerStreamingWrapper(nn.Module):
    """Simplified layer streaming wrapper."""
    
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        active_count: int = 1,  # number of simultaneously active layers
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        self._active_count = active_count
        self._store = _SimpleLayerStore(self._layers, self._target_device)
        
        # Move non-layer parameters to GPU
        self._move_non_layer_params_to_gpu()
        
        # Register hooks
        self._register_simple_hooks()
    
    def _move_non_layer_params_to_gpu(self) -> None:
        """Move non-layer parameters to GPU."""
        layer_tensor_ids = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)
    
    def _register_simple_hooks(self) -> None:
        """Register simple load/unload hooks."""
        idx_map = {id(layer): idx for idx, layer in enumerate(self._layers)}
        
        def _pre_hook(module: nn.Module, input, *, idx: int):
            # Load the current layer to GPU
            self._store.load_layer_to_gpu(idx, module)
            # Record the stream to prevent memory from being reclaimed prematurely
            for param in itertools.chain(module.parameters(), module.buffers()):
                param.data.record_stream(torch.cuda.current_stream(self._target_device))
        
        def _post_hook(module: nn.Module, input, output, *, idx: int):
            # Immediately move the layer back to CPU after processing
            self._store.unload_layer_from_gpu(idx, module)
        
        for layer in self._layers:
            idx = idx_map[id(layer)]
            pre_hook = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            post_hook = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying model."""
        try:
            # First try to get the attribute from the wrapper itself
            return super().__getattr__(name)
        except AttributeError:
            # If that fails, get it from the underlying model
            return getattr(self._model, name)
    

