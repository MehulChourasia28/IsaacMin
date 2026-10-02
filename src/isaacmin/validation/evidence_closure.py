"""Recursive byte verification for explicitly hash-bound numerical evidence."""
from pathlib import Path
import json
from .global_intersections import _verify_record


def verify_json_evidence(path,*,maximum_files=20000):
    """Follow only recorded path+sha256 relationships, never instructions."""
    seen=set()
    def visit_file(record,parent):
        record=dict(record);path=Path(record['path']);path=path if path.is_absolute() else parent/path;record['path']=str(path.resolve())
        key=(record['path'],record['sha256'])
        if key in seen:return
        if len(seen)>=maximum_files:raise ValueError('Bounded evidence closure exceeded')
        _verify_record(record);seen.add(key)
        if path.suffix=='.json':visit(json.loads(path.read_text()),path.parent)
    def visit(value,parent):
        if isinstance(value,dict):
            if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):visit_file(value,parent)
            else:
                for key,child in value.items():
                    base=Path(value['source_path']) if value.get('kind')=='WorldIR' and key=='source_dependency_closure' else parent
                    visit(child,base)
        elif isinstance(value,list):
            for child in value:visit(child,parent)
    path=Path(path).resolve();visit(json.loads(path.read_text()),path.parent)
    return len(seen)
