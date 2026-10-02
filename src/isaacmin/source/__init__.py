"""Read-only Minecraft Java source conversion."""
from .inventory import inspect_source
from .snapshot import snapshot_world, verify_source_unchanged
from .world_ir import extract_world_ir
from .nbt import SourceError
from .macro import extract_macro_surface

__all__ = ["inspect_source", "snapshot_world", "verify_source_unchanged", "extract_world_ir", "extract_macro_surface", "SourceError"]
