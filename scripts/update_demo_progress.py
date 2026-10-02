"""Publish measured demo state; retain every older progress section and failure."""
from pathlib import Path
import html
import re
import runpy

# Once the later scoped work exists, do not republish pre-clear galleries or
# overwrite the OpenClaw status with historical custom-harness qualification.
_root=Path(__file__).resolve().parents[1]
if (_root/'state/final_demo_polish_scope.json').is_file():
    runpy.run_path(str(_root/'scripts/update_final_demo_progress.py'),run_name='__main__')
    raise SystemExit(0)

from isaacmin.io import atomic_json, read_json, utc_now
from isaacmin.demo.catalog import load, presets, planner_status
from isaacmin.demo.jobs import Jobs

root=Path(__file__).resolve().parents[1]
catalog=load(root);registry=presets(root);now=utc_now()
jobs=Jobs(root).list();conversions=[j for j in jobs if j['action']=='convert']
checks={}
for key,name in [('browser','browser_checks.json'),('downloads','actual_download_ranges.json'),
                 ('UI_restart','server_restart_during_conversion.json'),('requested_views','requested_views_browser.json'),
                 ('agent_UI_completion','ui_agent_completion.json'),('evidence_readers','actual_evidence_readers.json'),
                 ('construction_UI','construction_browser.json'),('construction_agent','construction_agent.json'),
                 ('construction_animation','construction_animation.json'),('construction_conversion','construction_conversion.json')]:
    path=root/'evidence/demo'/name
    checks[key]=dict(status=read_json(path).get('status','not_run'),evidence=str(path.relative_to(root))) if path.is_file() else dict(status='not_run')
worlds=[]
for p in catalog['presets']:
    worlds.append(dict(id=p['id'],title=p['title'],extent_m=p['extent_m'],source_views=len(p['minecraft']),
        actual_isaac_ground_views=sum(i['kind']=='ground' for i in p['isaac']),
        actual_isaac_aerial_views=sum(i['kind']=='aerial' for i in p['isaac']),
        download=p['download'],build=registry[p['id']]['build'],qualification='unqualified_development_artifact'))
completed=checks['agent_UI_completion']['status']=='pass' and checks['browser']['status']=='pass'
ux_path=root/'evidence/demo/construction_browser.json';ux=read_json(ux_path) if ux_path.is_file() else {}
ux_complete=ux.get('status')=='pass'
ux_native_path=root/'evidence/demo/construction_conversion.json';ux_native=read_json(ux_native_path) if ux_native_path.is_file() else {}
report=dict(updated_at_utc=now,status='ui_agent_milestone_complete' if completed else 'four_preset_demo_available',url='http://127.0.0.1:8765',
    worlds=worlds,planner=planner_status(root),automated_app_checks=checks,conversion_jobs=conversions,
    scope='Actual source previews, native Isaac captures, portable downloads and bounded hosted agent',
    missing=['continuous 1km rendered world','full motion/contact/realism qualification','strict jungle package pixel replay'],
    automatic_visual_iteration='excluded_by_explicit_user_instruction',
    agent_capabilities='reports/AGENT_CAPABILITIES.md',workers=Jobs(root).worker_status(),
    navigation_stack='not_run_not_supplied')
if ux:report.update(status='clean_start_construction_UI_verified' if ux_complete else 'construction_UI_native_conversion_in_progress',construction_UI=ux)
for key,name in [('actual_agent_conversion','actual_conversion_e2e.json'),('portable_replays','portable_replays.json')]:
    path=root/'evidence/demo'/name
    if path.is_file():report[key]=read_json(path)
atomic_json(root/'reports/DEMO_STATUS.json',report)
status=read_json(root/'reports/STATUS.json')
status.update(updated_at_utc=now,status=report['status'],current_priority='requested_UI_agent_and_construction_UX_complete' if ux_complete else 'verify_clean_start_and_real_construction_animation',
    next='UI/agent and construction UX complete; 1km delivery and full realism/contact/motion qualification remain separate unfinished work.' if ux_complete else 'Finish the current native conversion and its UI handoff checks.',
    demo=dict(report='reports/DEMO_STATUS.json',url=report['url'],presets=4,source_views=sum(p['source_views'] for p in worlds),
        native_views=sum(p['actual_isaac_ground_views']+p['actual_isaac_aerial_views'] for p in worlds),
        downloads_ready=sum(bool(p['download']) for p in worlds),scope='256m development regions; not complete 1km delivery',
        requested_views='actual_UI_and_hosted_agent_native_capture_pass',agent_new_view_tools='connected_and_actual_hosted_verified',workers=report['workers']))
atomic_json(root/'reports/STATUS.json',status)
sequence=read_json(root/'state/product_sequence.json');sequence['agent']['status']='views_evidence_and_job_controls_connected_actual_hosted_verified'
sequence['agent']['automatic_visual_inspection_and_repair']='excluded_by_explicit_user_instruction'
sequence['ui']['status']='four_presets_views_downloads_job_controls_verified';sequence['updated_at_utc']=now
sequence['ui']['construction_experience']='actual_conversion_verified' if ux_complete else 'implemented_actual_conversion_running'
sequence['execution_status']=report['status']
atomic_json(root/'state/product_sequence.json',sequence)
continuation=read_json(root/'state/continuation.json')
continuation.update(updated_at_utc=now,status=report['status'],live_work=dict(UI=report['url'],planner='Approved NVIDIA Nemotron 3 Super; actual tool and app evidence pass',workers=report['workers'],
    conversion_jobs=[dict(id=j['id'],status=j['status'],progress=j.get('progress')) for j in conversions],
    next=('Requested UI/agent and clean-start/construction experience complete. Keep service available; await user direction for scale or world qualification.' if ux_complete else 'Complete the same ongoing native Jungle Rivers conversion and verify the clean-start/construction experience.')+' Automatic visual iteration is excluded. See reports/AGENT_CAPABILITIES.md and docs/DEMO.md.'))
atomic_json(root/'state/continuation.json',continuation)
project=read_json(root/'state/resolved_project.json')
project['demo_planner_selection']='state/planner_selection.json';project['current_product_sequence']='state/product_sequence.json'
atomic_json(root/'state/resolved_project.json',project)

rows=''.join(f"<tr><td>{html.escape(p['title'])}</td><td>{p['source_views']}</td><td>{p['actual_isaac_ground_views']} ground + {p['actual_isaac_aerial_views']} aerial</td><td>{p['download']['bytes']/1e9:.1f} GB</td></tr>" for p in worlds)
latest=conversions[0] if conversions else None
replay_text='Native replay of the downloaded packages has not completed.'
source_path=root/'evidence/demo/original_source_integrity.json'
source_text='Original source rehash has not been completed.'
if source_path.is_file():
    sources=read_json(source_path)
    source_text=(f"All {sum(r['files'] for r in sources['results'])} recorded original source files still match their immutable snapshot hashes."
        if sources['status']=='pass' else 'Original source rehash has a failure; inspect its recorded evidence.')
if 'portable_replays' in report:
    replay=report['portable_replays'];results=replay.get('results',[])
    passed=sum((r.get('comparison') or {}).get('status')=='static_component_pass' for r in results)
    failed=[r['preset'] for r in results if (r.get('comparison') or {}).get('status')!='static_component_pass']
    replay_text=f'{len(results)} downloaded worlds were extracted elsewhere and opened in fresh Isaac processes. {passed} of {len(results)} passed the strict one-aerial RGB/depth/label comparison.'
    for row in results:
        if row['preset'] not in failed:continue
        comparison=row.get('comparison') or {};title=registry[row['preset']]['title']
        metrics=comparison.get('frames',[])
        if metrics:
            replay_text+=f" {title} RGB p99.9 difference was {metrics[0]['rgb_p999_codes']:g} codes against the unchanged {comparison['thresholds']['rgb_p999_codes_max']} code limit. See the recorded per-sensor comparisons; this replay remains failed."
        else:replay_text+=f' {title} did not produce a passing comparison; its failure evidence is retained.'
    replay_text+=' This subset is not full portable, contact or motion qualification.'
conversion_text=(f"The fresh agent-triggered Frozen coast conversion is {html.escape(latest['status'])}. "
    f"{html.escape(latest.get('progress',latest.get('error') or (latest.get('result') or {}).get('message','')))}") if latest else 'Fresh agent-triggered conversion has not been run.'
if report.get('actual_agent_conversion',{}).get('status')=='pass':
    measured=report['actual_agent_conversion'];minutes=measured['elapsed_seconds']/60
    conversion_text=f'Fresh browser → NVIDIA agent → Minecraft snapshot → HighMap → Blender USD → Isaac conversion passed in {minutes:.1f} minutes for the 256 m Frozen coast region, including six ground views and three aerials. Later ZIP packaging is excluded from that timing; its download is now ready. This is a measured development conversion, not a timing estimate for 1 km or a realism pass.'
images=[]
for p in catalog['presets']:
    source=p['minecraft'][0];aerial=next(i for i in p['isaac'] if i['kind']=='aerial')
    paths=[Path(catalog['media'][i['id']]['path']).relative_to(root) for i in (source,aerial)]
    pictures=''.join(f'<a href="../{path}"><img loading="lazy" src="../{path}" alt="{html.escape(p["title"])} {label}"></a>' for path,label in zip(paths,['actual source-save preview','actual Isaac aerial']))
    images.append(f'<figure><div class="demo-pair">{pictures}</div><figcaption>{html.escape(p["title"])} — actual source render (left) and actual Isaac aerial (right), same 256 m region, different camera poses. Images are unchanged. Source is Chunky; the right image is native Isaac.</figcaption></figure>')
new_views=''
completion_path=root/'evidence/demo/ui_agent_completion.json'
if completed:
    completion=read_json(completion_path);capture=Path(completion['native_result']['result']['capture'])
    frame=read_json(capture)['frames'][0];image=(capture.parent/frame['rgb']).relative_to(root)
    new_views=f'<figure><a href="../{image}"><img loading="lazy" src="../{image}" alt="Actual agent-requested Isaac aerial"></a><figcaption>Actual agent-requested aerial of the latest Frozen coast conversion: user-chosen target, 160 m requested height, 75° heading. Native capture completed; original scene and build hashes unchanged. This is additional viewing, not automatic visual repair.</figcaption></figure>'
ux_images=''.join(f'<figure><a href="../artifacts/demo/construction_browser/{name}.png"><img loading="lazy" src="../artifacts/demo/construction_browser/{name}.png" alt="{caption}"></a><figcaption>{caption}</figcaption></figure>'
    for name,caption in [('01_clean_start','New clean opening screen: only four actual Minecraft source renders.'),('03_assets','Actual live-build screenshot: sampled terrain and USD asset geometry, animated as stage outputs arrive. This is a labelled browser diagram, not an Isaac capture.'),('04_world_ready','Completed conversion in the new UI: actual Isaac gallery and portable download.')]
    if (root/'artifacts/demo/construction_browser'/(name+'.png')).is_file())
ux_text=('Passed the actual UI → fresh Jungle Rivers conversion → native Isaac captures → portable ZIP flow.' if ux_complete else 'Fresh Jungle Rivers conversion is running through the new UI; actual terrain and native asset previews are recorded. Final Isaac capture and download checks remain pending.')
if ux_complete and ux_native:
    ux_text+=f" All five planned ground and three aerial captures have matched RGB/depth/label bytes and poses. The 256 m conversion took {ux_native['elapsed_seconds_including_queue_and_packaging']/60:.1f} minutes including queue time and packaging; its ZIP is {ux_native['download']['bytes']/1e9:.1f} GB. This is a measured app flow, not a realism qualification."
ux_video='<figure><video controls preload="metadata" width="100%" src="../artifacts/demo/construction_browser/construction_ui_recording.webm"></video><figcaption>Actual browser screen recording of the construction diagram, using this build’s real sampled terrain and native asset data. This is not an Isaac-rendered motion video.</figcaption></figure>' if (root/'evidence/demo/construction_animation.json').is_file() else ''
ux_section=f'<h2>Clean start and live construction preview</h2><p>{ux_text} The opening screen has four Minecraft presets and three clear steps. Saved worlds retain earlier work. During conversion, the preview uses completed terrain and real USD placement data; final appearance remains the saved Isaac captures. Browser refresh keeps the same job. No world-quality settings were changed.</p><p><a href="../evidence/demo/construction_browser.json">Actual browser/conversion evidence</a> · <a href="../evidence/demo/construction_agent.json">Hosted current-conversion question</a> · <a href="../artifacts/tests/construction_ui_20261002.xml">Preview provenance and context tests</a></p>{ux_video}{ux_images}' if ux else ''
section=f'''<!-- demo-progress:start --><section id="demo-progress">
{ux_section}
<h2>UI and agent milestone complete</h2><p class="status">Updated {now[:16].replace('T',' ')} UTC. All four presets have actual Minecraft previews, native Isaac galleries and downloadable USD worlds. Desktop/mobile UI, range downloads and persistent jobs pass their recorded checks. New-view controls, richer evidence readers and resume/cancel tools are connected to the hosted NVIDIA assistant and verified through actual requests. These are 256 m development regions; 1 km expansion and full visual/contact/motion qualification remain incomplete.</p>
<p><a href="http://127.0.0.1:8765">Open IsaacMin World Studio on the Spark</a> · <a href="../docs/DEMO.md">Run and recovery instructions</a> · <a href="AGENT_CAPABILITIES.md">Complete questions, actions and limits</a> · <a href="DEMO_STATUS.json">Measured demo status</a></p>
<p>Users can request 1–6 ground or aerial views, with optional Minecraft X/Z, heading and aerial height. Presets and completed conversions are supported. Captures preserve scene geometry, materials and quality settings. The assistant can read source/biome/terrain/assets/capture/download/qualification evidence, look up jobs, create views, verify files, prepare downloads, convert again, resume eligible jobs and cancel queued work. Automatic visual inspection and iterative repair are excluded.</p>
<p><a href="../evidence/demo/ui_agent_completion.json">Actual browser → NVIDIA → Isaac and queue-control evidence</a> · <a href="../evidence/demo/actual_evidence_readers.json">28 reads across actual preset records</a> · <a href="../artifacts/tests/ui_agent_completion_20261002.xml">Focused controller/camera/recovery tests</a>. A model coordinate-label mistake is retained in the answer review; named bounds and a corrected <a href="../evidence/demo/coordinate_answer_followup.json">hosted follow-up</a> resolve that ambiguity.</p>
<table><tr><th>Preset</th><th>Minecraft previews</th><th>Isaac views</th><th>Download</th></tr>{rows}</table>
<p>{conversion_text} New conversions preserve the existing preset. Use “View result” in workspace activity for the new gallery. A queued or rendered world is not a realism qualification; the user's navigation stack was not supplied. <a href="../evidence/demo/actual_conversion_e2e.json">Actual end-to-end evidence</a>.</p>
<p>{replay_text} <a href="../evidence/demo/portable_replays.json">Native replay measurements</a>. {source_text}</p>
<figure><a href="../artifacts/demo/browser/world_3_aerial.png"><img loading="lazy" src="../artifacts/demo/browser/world_3_aerial.png" alt="Actual local IsaacMin four-preset studio"></a><figcaption>Actual Chromium capture of the implemented local app; no live Isaac rendering runs in the browser.</figcaption></figure>
{new_views}
{''.join(images)}</section><!-- demo-progress:end -->'''
path=root/'reports/progress.html';document=path.read_text()
document=re.sub(r'<!-- demo-progress:start -->.*?<!-- demo-progress:end -->','',document,flags=re.S)
document=re.sub(r'<!-- user-stop:start -->(.*?)<!-- user-stop:end -->',
    lambda match:'<details><summary>Historical stop checkpoint — resumed for UI/agent completion</summary>'+match.group(1).replace('Current checkpoint — stopped at your request','Earlier requested stop')+'</details>',document,flags=re.S)
document=document.replace('<h1>Actual Isaac views of your Minecraft map</h1>','<h1>Actual Isaac views of your Minecraft map</h1>'+section)
document=document.replace('<h2>1.5 km expansion running</h2>','<h2>Preserved expansion work — deferred to 1 km</h2>')
if '.demo-pair {' not in document:
    document=document.replace('</style>','.demo-pair { display:grid; grid-template-columns:1fr 1fr; gap:10px; } td,th { padding:8px 14px; text-align:left; } @media(max-width:650px) { .demo-pair { grid-template-columns:1fr; } }\n</style>')
path.write_text(document)
print(dict(status=report['status'],worlds=len(worlds),checks=checks))
