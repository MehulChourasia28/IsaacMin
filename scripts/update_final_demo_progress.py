"""Publish the scoped OpenClaw/visual work without rewriting historical evidence."""
from datetime import datetime
import html
import re
from pathlib import Path

from isaacmin.io import atomic_json, read_json, sha256_file, utc_now

root=Path(__file__).resolve().parents[1]
def read(relative):
    path=root/relative
    return read_json(path) if path.is_file() else {}
def link(relative,label):
    return f'<a href="../{html.escape(relative)}">{html.escape(label)}</a>'
def figure(relative,caption):
    return f'<figure><a href="../{relative}"><img loading="lazy" src="../{relative}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'

visual=read('evidence/demo/agent_pipeline_browser.json')
pipeline=read('evidence/demo/openclaw_pipeline_browser.json')
chat=read('evidence/demo/openclaw_chat_browser.json')
ready=visual.get('status')=='pass' and pipeline.get('status')=='pass' and chat.get('status')=='pass'
now=utc_now()
visual_id=visual.get('job_id')
baseline='artifacts/demo/jobs/job_fcc81a2b268c42f5a0be6a64a45d8925/build/assembled'
current=f'artifacts/demo/jobs/{visual_id}/build/assembled'
images={}
for group in ('preview','overview'):
    for path in sorted((root/current/group).glob('rgb_*.png')):
        images[str(path.relative_to(root))]=sha256_file(path)
population=read(f'artifacts/demo/jobs/{visual_id}/build/outdoor_build.json').get('stages',{}).get('population',{})
visual_report=dict(status='isaac_captures_verified' if visual.get('status')=='pass' else 'in_progress',
    job_id=visual_id, images=images, image_processing='none; original Isaac PNGs',
    changes=dict(daylight_lift_ev_max=.45,shade_response='lift fades to zero in deep shade',
        grass='bounded growing-season pigment adjustment in native MDL; original texture and geometry retained',
        forest_floor='tropical ground-herb density doubled within existing ecological constraints',
        actual_population_instances=population.get('instances')),
    evidence='evidence/demo/agent_pipeline_browser.json',
    qualification='development visuals inspected; full realism, temporal and contact qualification not inferred')
atomic_json(root/'evidence/demo/lively_visuals_20261002.json',visual_report)
report=dict(updated_at_utc=now,status='requested_scope_complete' if ready else 'openclaw_native_verification_running',
    harness='OpenClaw',harness_version='2026.9.7',model='nvidia/nemotron-3-super-120b-a12b',
    pipeline=dict(status=pipeline.get('status','not_run'),job_id=pipeline.get('job_id'),
        evidence='evidence/demo/openclaw_pipeline_browser.json'),
    assistant=dict(status=chat.get('status','not_run'),view_job=chat.get('view_job'),
        evidence='evidence/demo/openclaw_chat_browser.json'),
    visuals=visual_report,library_clear=read('evidence/demo/library_cleared_20261002.json'),
    render_optimization='stopped_by_user; experimental changes not selected; production quality unchanged',
    automatic_visual_iteration='excluded_by_user',stop_after_requested_work=True,
    missing=['continuous 1km delivery','full realism, motion and contact qualification',
        'clean-package replay of the latest visual changes'],navigation_stack='not_run_not_supplied')
atomic_json(root/'reports/FINAL_DEMO_POLISH.json',report)
demo=read('reports/DEMO_STATUS.json')
demo.update(updated_at_utc=now,status=report['status'],current_harness='OpenClaw',
    current_scope_report='reports/FINAL_DEMO_POLISH.json',
    saved_library_policy='Earlier saved worlds archived by user request; new conversions remain in activity',
    historical_worlds_notice='Historical worlds below describe pre-clear artifacts; consult the UI catalogue for visible worlds.')
atomic_json(root/'reports/DEMO_STATUS.json',demo)
scope=read('state/final_demo_polish_scope.json')
scope.update(at_utc=now,scope='OpenClaw setup and livelier visuals; no further render optimisation; stop when verified',
    status=report['status'],agent=report['pipeline'],visual_changes=visual_report,
    openclaw_version='2026.9.7',job_id=pipeline.get('job_id'),stop_after_requested_work=True)
atomic_json(root/'state/final_demo_polish_scope.json',scope)
status=read('reports/STATUS.json');status.update(updated_at_utc=now,final_demo_polish=report,
    current_priority='Requested OpenClaw/visual scope completed; stopped' if ready else 'Verify OpenClaw conversion and requested Isaac view')
atomic_json(root/'reports/STATUS.json',status)

agent_text=('The actual Create button → OpenClaw/NVIDIA → all eleven ordered stages → '
    'Isaac gallery → portable ZIP check passed.' if pipeline.get('status')=='pass' else
    'The real Create button is currently running a fresh Forest valley conversion through OpenClaw. '
    'Completion and download verification remain pending.')
view_text=('An actual browser request through OpenClaw created an additional Isaac aerial at '
    '180 m height and 120° heading in the existing Jungle world, and it appeared in the correct gallery.'
    if chat.get('status')=='pass' else 'OpenClaw has accepted a real request for an extra Jungle aerial; native capture/gallery verification is pending.')
elapsed=''
if visual.get('completed_at_utc'):
    seconds=(datetime.fromisoformat(visual['completed_at_utc'].replace('Z','+00:00'))-
             datetime.fromisoformat(visual['started_at_utc'].replace('Z','+00:00'))).total_seconds()
    elapsed=f'The completed refreshed Jungle conversion took {seconds/60:.1f} minutes including its eight captures and ZIP. '
paired=''
for suffix,caption in [('preview/rgb_00000.png','Ground view: earlier appearance (left), refreshed native grass and daylight (right).'),
                       ('preview/rgb_00001.png','Forest floor: earlier (left), more living ground cover (right).')]:
    paired+=f'<figure><div class="demo-pair"><img loading="lazy" src="../{baseline}/{suffix}" alt="Earlier Isaac view"><img loading="lazy" src="../{current}/{suffix}" alt="Updated Isaac view"></div><figcaption>{caption} Same planned camera location; original unedited Isaac images.</figcaption></figure>'
forest=''
forest_path=f'artifacts/demo/jobs/{pipeline.get("job_id")}/build/assembled/preview/rgb_00000.png'
if (root/forest_path).is_file():
    forest=figure(forest_path,'Forest valley: new native Isaac ground view from the OpenClaw-controlled conversion. The same growing-season material and daylight rules apply across both real maps.')
section=f'''<!-- final-demo-polish:start --><section id="final-demo-polish">
<h2>OpenClaw agent and livelier Isaac worlds</h2>
<p class="status">Updated {now[:16].replace('T',' ')} UTC. <strong>{'Requested work complete; development stopped.' if ready else 'Final native verification in progress.'}</strong> The assistant now runs on OpenClaw 2026.9.7 with NVIDIA Nemotron. Greener native grass, brighter daylight and denser tropical ground cover are implemented and visible in the actual Isaac world.</p>
<p><a href="http://127.0.0.1:8765">Open World Studio</a> · <a href="AGENT_CAPABILITIES.md">All supported questions and actions</a> · <a href="../docs/DEMO.md">Operation and recovery</a> · <a href="FINAL_DEMO_POLISH.json">Current machine-readable report</a></p>
<p>{agent_text} {view_text} The controller permits only the next stage and exact checkpoint; fixed source, quality settings and native completion receipts remain authoritative. OpenClaw has only the typed IsaacMin tools, and receives no NVIDIA key. Original custom-harness sessions are retained as historical evidence.</p>
<p>{link('evidence/demo/openclaw_pipeline_browser.json','Actual full pipeline check')} · {link('evidence/demo/openclaw_chat_browser.json','Actual assistant/new-view check')} · {link('artifacts/tests/openclaw_transport_final_20261002.xml','Transport and stage boundary fixtures')} · {link('configs/openclaw/runtime.lock.json','Pinned ARM64 runtime')}</p>
<p>The old saved library was cleared: 29 previous jobs and four historical galleries/downloads are hidden from the UI; original files remain as evidence. The home page starts with four Minecraft source choices. New verification conversions appear separately in Saved worlds.</p>
<p>{elapsed}Rendering-time optimisation was dropped at your request; the original capture worker and sample count remain selected. No speed improvement is claimed. {link('evidence/demo/lively_visuals_20261002.json','Visual changes and original image hashes')}.</p>
{paired}
{forest}
{figure(current+'/overview/rgb_00001.png','Updated Jungle aerial: actual native Isaac capture of the 256 m region. Surface geography and water remain source-derived.')}
<p>These are 256 m development worlds. Continuous 1 km delivery and full realism/motion/contact qualification remain unfinished. The latest visual changes have not received a fresh extracted-package pixel replay. Earlier replay failures are retained below. The user’s navigation stack was not supplied or tested. Automatic visual inspection/repair remains excluded.</p>
</section><!-- final-demo-polish:end -->'''
path=root/'reports/progress.html';document=path.read_text()
pattern=r'<!-- final-demo-polish:start -->.*?<!-- final-demo-polish:end -->'
if re.search(pattern,document,re.S):document=re.sub(pattern,lambda _:section,document,flags=re.S)
else:
    old=r'<!-- demo-progress:start -->.*?<!-- demo-progress:end -->'
    document=re.sub(old,lambda m:'<details><summary>Earlier UI report — before library clearing and OpenClaw migration</summary>'+m[0]+'</details>',document,flags=re.S)
    document=document.replace('</h1>','</h1>'+section,1)
path.write_text(document)
print(report['status'])
if (root/'reports/REPOSITORY_CLEANUP.json').is_file():
    import runpy
    runpy.run_path(str(root/'scripts/update_cleanup_progress.py'),run_name='__main__')
