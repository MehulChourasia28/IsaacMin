"""Bind standalone geometry checks to the exact declared source payloads."""
from __future__ import annotations

import json
from pathlib import Path

from ..io import sha256_file
from ..security import safe_path


def verify_source_topology(ir_dir):
    """Verify WorldIR and graph payload bytes before any labels are consumed.

    The manifest is the authority supplied by the caller. This detects altered
    or stale payloads; it does not claim an external signature or source trust.
    """
    ir_dir=Path(ir_dir)
    ir_path=ir_dir/"world_ir.json"
    graph_path=ir_dir/"topology/source_topology_graph.json"
    ir=json.loads(ir_path.read_text());graph=json.loads(graph_path.read_text())
    ir_hash=sha256_file(ir_path)
    if graph.get("source_ir_sha256")!=ir_hash:
        raise ValueError("Source topology graph belongs to a different WorldIR")
    payloads=[]
    for root,entries,label in ((ir_dir,ir.get("files",[]),"IR"),
                               (graph_path.parent,graph.get("files",[]),"topology")):
        names=[entry["path"] for entry in entries]
        if len(names)!=len(set(names)):
            raise ValueError("Source manifest contains duplicate payload paths")
        if label=="topology" and "source_topology_labels.npz" not in names:
            raise ValueError("Source graph does not bind its topology labels")
        for entry in entries:
            path=safe_path(root,entry["path"],must_exist=True)
            digest=sha256_file(path);size=path.stat().st_size
            if digest!=entry["sha256"] or ("bytes" in entry and size!=entry["bytes"]):
                raise ValueError(f"Source {label} payload changed")
            if label=="topology" and "bytes" not in entry:
                raise ValueError("Source topology payload lacks its declared size")
            payloads.append({"path":str(path.resolve()),"sha256":digest,"bytes":size})
    context={"source_ir_sha256":ir_hash,"source_topology_sha256":sha256_file(graph_path),
             "payloads":payloads,"verifier_sha256":sha256_file(Path(__file__))}
    return graph,context


def recheck_source_topology(ir_dir,context):
    """Prevent evidence publication if source inputs changed during a check."""
    if verify_source_topology(ir_dir)[1]!=context:
        raise ValueError("Source evidence changed during geometry validation")
