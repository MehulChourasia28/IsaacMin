"""Publish only existing, unaltered Isaac captures to the progress HTML."""
from pathlib import Path
import argparse
import html
import json
import re
from datetime import datetime,timezone


def update(root,builds):
    sections=[];records=[]
    for directory in builds:
        directory=directory.resolve();manifest=directory/'outdoor_build.json'
        if not manifest.is_file():continue
        build=json.loads(manifest.read_text());images=[];captures=[]
        for relative in ('litter_inspection_1/capture','camera_inspection_1/capture','assembled/focus','assembled/preview','shoreline_inspection/capture','biome_inspection_1/capture','aerial_inspection_1/capture','assembled/overview'):
            capture=directory/relative
            result=capture/'preview_result.json'
            if not result.is_file():continue
            record=json.loads(result.read_text())
            captures.append(dict(path=str(capture.relative_to(root)),frames=len(record['frames']),
                renderer=record['renderer_recipe'],status=record['status']))
            selected=([0,1] if relative.endswith('focus') or 'biome_inspection' in relative or 'litter_inspection' in relative else [0,1] if relative.endswith('overview')
                      else [1] if 'frozen' in directory.name and len(record['frames'])>1 else [0])
            for i in selected:
                if i>=len(record['frames']):continue
                if relative=='assembled/focus' and i==0 and (directory/'camera_inspection_1/capture/preview_result.json').is_file():
                    corrected=json.loads((directory/'camera_inspection_1/capture/preview_result.json').read_text())
                    if corrected['scene_sha256']==record['scene_sha256'] and corrected['frames'][0]['pose']['position']==record['frames'][0]['pose']['position']:
                        continue
                image=capture/record['frames'][i]['rgb']
                if not image.is_file():raise ValueError('Capture result references missing RGB')
                caption=('Aerial view' if relative.endswith('overview') or 'aerial_inspection' in relative or (('biome_inspection' in relative or 'litter_inspection' in relative) and i==1) else 'Shoreline view' if 'shoreline' in relative else 'Ground view')
                images.append((image,caption))
        finding=directory/'appearance_findings.json'
        findings=json.loads(finding.read_text()) if finding.exists() else {'status':'not_qualified','defects':['Visual inspection pending; no realism qualification inferred.']}
        label=Path(build['source_world']).name;extent=build['identity']['extent']
        sections.append(f'<section><h3>{html.escape(label)} · {extent} × {extent} m · {html.escape(directory.name)}</h3>')
        sections.append('<p class="status">Real Isaac captures. Development status: '+html.escape(findings['status'])+'. '+html.escape(' '.join(findings.get('defects',[])))+'</p>')
        for image,caption in images:
            url='../'+image.relative_to(root).as_posix()
            caption+=f' of the actual {extent} m region. Unaltered native RGB; the surrounding HDRI is not reconstructed terrain.'
            sections.append(f'<figure><a href="{url}"><img loading="lazy" src="{url}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>')
        sections.append(f'<p><a href="../{manifest.relative_to(root).as_posix()}">Build and producer evidence</a></p></section>')
        records.append(dict(build=str(directory.relative_to(root)),world=label,captures=captures,findings=findings))
    target=root/'reports/progress.html';page=target.read_text()
    section='<section id="multiworld-renders"><h2>Latest actual multiworld renders</h2><p>Mountains, valleys and visible water are retained. Caves, ravines and underground structures are outside the requested scope. Focused visual iterations use two views with unchanged geometry, resolution and 1,024-sample rendering; full view coverage follows separately. No scene below is realism-qualified.</p>'+''.join(sections)+'</section>'
    marker=r'<!-- multiworld-renders:start -->.*?<!-- multiworld-renders:end -->'
    if not re.search(marker,page,re.S):raise ValueError('Existing gallery marker missing')
    target.write_text(re.sub(marker,'<!-- multiworld-renders:start -->'+section+'<!-- multiworld-renders:end -->',page,flags=re.S))
    (root/'reports/LATEST_OUTDOOR_GALLERY.json').write_text(json.dumps(dict(
        at_utc=datetime.now(timezone.utc).isoformat(),builds=records,qualification='not_inferred'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('builds',nargs='+',type=Path)
    a=p.parse_args();update(Path.cwd(),a.builds)
    print('Progress gallery updated from completed native capture records.')
