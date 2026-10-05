"""Compatibility shim for .venv-vllm (put on PYTHONPATH by local_common.run_worker only).

vLLM 0.30's pixtral.py imports transformers' old name PixtralRotaryEmbedding, renamed to
PixtralVisionRotaryEmbedding in transformers 5.18, and
position_ids_in_meshgrid was removed. Mistral3 checkpoints import pixtral.py even when run
text-only (the vision tower is never built), so alias the old name.
"""
try:
    import transformers.models.pixtral.modeling_pixtral as _mp
    if not hasattr(_mp, "PixtralRotaryEmbedding") and hasattr(_mp, "PixtralVisionRotaryEmbedding"):
        _mp.PixtralRotaryEmbedding = _mp.PixtralVisionRotaryEmbedding
    if not hasattr(_mp, "position_ids_in_meshgrid"):
        def position_ids_in_meshgrid(patch_embeds_list, max_width):     # transformers < 5.18 implementation
            import torch
            positions = []
            for patch in patch_embeds_list:
                height, width = patch.shape[-2:]
                mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
                h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
                positions.append((h_grid * max_width + v_grid)[:, 0])
            return torch.cat(positions)
        _mp.position_ids_in_meshgrid = position_ids_in_meshgrid
except Exception:
    pass


# vLLM 0.30's InternLM2ForCausalLM.forward (used by Bespoke-MiniCheck-7B) declares intermediate_tensors without a
# default, but the engine's profiling run calls it without that argument. Give it the default None (as every
# other model has) when vLLM imports the module; nothing is imported eagerly here.
import importlib.abc as _abc
import sys as _sys

_TARGET = "vllm.model_executor.models.internlm2"


class _PatchInternLM2(_abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        _sys.meta_path.remove(self)
        try:
            import importlib.util
            spec = importlib.util.find_spec(name)
        finally:
            _sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            cls = getattr(module, "InternLM2ForCausalLM", None)
            if cls is not None and not getattr(cls, "_shim_patched", False):
                fwd = cls.forward

                def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw):
                    return fwd(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kw)
                cls.forward = forward
                cls._shim_patched = True
        spec.loader.exec_module = exec_module
        return spec


_sys.meta_path.insert(0, _PatchInternLM2())
