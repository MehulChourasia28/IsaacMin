"""Versioned artifact contracts shared across process boundaries."""
from .artifacts import Artifact, FileRecord, ArtifactError, seal_directory, verify_artifact
from .coordinates import CoordinateFrame

__all__ = ["Artifact", "FileRecord", "ArtifactError", "CoordinateFrame", "seal_directory", "verify_artifact"]

