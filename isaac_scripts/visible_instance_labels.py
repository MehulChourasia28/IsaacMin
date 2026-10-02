"""Keep identities for every observed native pixel without global JSON expansion."""
import numpy as np


def visible_instance_labels(segments):
    info=segments.get('info',segments)
    pixels=np.asarray(segments['data'])
    if pixels.dtype!=np.uint32 or pixels.ndim!=2:raise ValueError('Expected native uint32 instance pixels')
    visible=np.unique(pixels);result={}
    if 'idToLabels' in info:
        # Pinned Annotator.get_data converts the native IDs/token arrays into
        # this mapping even for the fast node. Filter its actual values; never
        # assume that avoiding the legacy node also avoids this SDK conversion.
        labels=info['idToLabels']
        for ident in visible:
            key=int(ident);value=labels.get(key,labels.get(str(key)))
            if value is not None:result[str(key)]=value
        missing=[int(i) for i in visible if int(i)>1 and str(int(i)) not in result]
        if missing:raise ValueError('Native segmentation pixels lack identity metadata')
        return result,dict(native_entries=len(labels),retained_visible_entries=len(result),
            policy='Every observed native pixel identity; unseen report entries omitted',
            annotator='instance_id_segmentation_fast',
            adapter='Pinned SDK get_data ID mapping; full native mapping remains in memory during readback')
    ids=np.asarray(info['ids']).reshape(-1);labels=info['labels']
    if len(ids)!=len(labels):raise ValueError('Native instance ID/path arrays disagree')
    for start in range(0,len(ids),250000):
        values=ids[start:start+250000];where=np.searchsorted(visible,values)
        keep=where<len(visible);where=np.minimum(where,len(visible)-1)
        keep&=visible[where]==values
        for offset in np.flatnonzero(keep):
            key=str(int(values[offset]));value=str(labels[start+offset])
            if key in result and result[key]!=value:raise ValueError('Native identity maps to conflicting paths')
            result[key]=value
    missing=[int(i) for i in visible if int(i)>1 and str(int(i)) not in result]
    if missing:raise ValueError('Native segmentation pixels lack identity metadata')
    return result,dict(native_entries=len(ids),retained_visible_entries=len(result),
        policy='Every ID present in the unchanged native pixel buffer; unobserved identity metadata omitted',
        annotator='instance_id_segmentation_fast',
        adapter='Pinned legacy annotator wraps this same data pointer and IDs/labels arrays in global JSON; use native arrays directly')
