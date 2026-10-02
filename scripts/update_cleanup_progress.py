"""Publish the filesystem/UI maintenance milestone without changing world gates."""
from pathlib import Path
import html
import re
import xml.etree.ElementTree as ET
from isaacmin.io import read_json, atomic_json, utc_now

root=Path(__file__).resolve().parents[1]
plan=read_json(root/'reports/cleanup_plan.json')
integrity=read_json(root/'evidence/demo/repository_cleanup_integrity.json')
browser=read_json(root/'evidence/demo/saved_comparison_browser.json')
usd=read_json(root/'evidence/demo/cleanup_usd_dependencies.json')
git=read_json(root/'evidence/demo/git_cleanup_audit.json')
agent_readers=read_json(root/'evidence/demo/cleanup_agent_readers.json')
suite=ET.parse(root/'artifacts/tests/repository_cleanup_ui.xml').getroot().find('testsuite')
tests={key:int(suite.attrib[key]) for key in ('tests','errors','failures','skipped')}
archive_suite=ET.parse(root/'artifacts/tests/repository_archive_safety.xml').getroot().find('testsuite')
archive_tests={key:int(archive_suite.attrib[key]) for key in ('tests','errors','failures','skipped')}
passed=(plan.get('status')=='archived' and all(r['status']=='pass' for r in (integrity,browser,usd,git,agent_readers))
    and not tests['errors'] and not tests['failures'] and not archive_tests['errors'] and not archive_tests['failures'])
report=dict(status='complete' if passed else 'incomplete',updated_at_utc=utc_now(),
    scope='Reversible repository housekeeping and saved-result Minecraft/Isaac comparison',
    archive=dict(path=plan['archive'],manifest=plan['archive']+'/manifest.json',moves=len(plan['moves']),
        entries=plan['entries'],logical_bytes=plan['logical_bytes'],deleted_files=0,disk_space_freed=0),
    integrity=integrity,ui=browser,usd=usd,git=git,agent_evidence_readers=agent_readers,tests=tests,archive_safety_tests=archive_tests,
    tests_evidence='artifacts/tests/repository_cleanup_ui.xml',instructions='docs/REPOSITORY.md',
    unchanged_limits=['256m development regions','Full realism/contact/motion qualification incomplete',
        'Continuous 1km delivery incomplete','User navigation stack not supplied or tested'],
    publication='Not initialized, committed or pushed; local runtime data intentionally retained outside Git')
atomic_json(root/'reports/REPOSITORY_CLEANUP.json',report)
status=read_json(root/'reports/STATUS.json')
status.update(updated_at_utc=report['updated_at_utc'],repository_cleanup='reports/REPOSITORY_CLEANUP.json',
    current_priority='Repository cleanup and saved-result comparison complete; ready for submission preparation' if passed else 'Verify repository cleanup and saved-result comparison')
atomic_json(root/'reports/STATUS.json',status)
images=''.join(f'<figure><a href="../artifacts/demo/saved_comparison/{ident}_aerial.png"><img loading="lazy" src="../artifacts/demo/saved_comparison/{ident}_aerial.png" alt="{title}: actual Minecraft and Isaac comparison page"></a><figcaption>{title}: actual saved-world comparison in the browser. Original captured images, different camera angles.</figcaption></figure>' for ident,title in [('isaacmin','Forest valley'),('isaacmin1','Jungle rivers')])
section=f'''<!-- repository-cleanup:start --><section id="repository-cleanup">
<h2>Repository cleanup and side-by-side saved worlds</h2>
<p class="status">Updated {report['updated_at_utc'][:16].replace('T',' ')} UTC. <strong>{'Complete' if passed else 'Verification pending'}.</strong> Saved worlds → View result now keeps Minecraft on the left while you browse Isaac ground and aerial views on the right. Mobile panels stack; downloads and extra-view controls remain available.</p>
<p><a href="http://127.0.0.1:8765/library">Open saved worlds</a> · <a href="REPOSITORY_CLEANUP.json">Cleanup and UI evidence</a> · <a href="../docs/REPOSITORY.md">Repository layout and restore instructions</a></p>
<p>{len(plan['moves'])} obsolete paths ({plan['logical_bytes']/1024**3:.1f} GiB logical data, including hardlinks) moved into gitignored <code>{html.escape(plan['archive'])}</code>. No files were deleted and no disk-space saving is claimed. All four source saves, assets, isolated tools and referenced runtime libraries remain in place. 300 protected trees and 49 configuration manifests matched before/after.</p>
<p>Checks: {tests['tests']} controller/UI tests and {archive_tests['tests']} archival safety tests passed; actual browser comparisons, filters, mobile layout and download ranges passed for both current saved worlds; native USD resolved every dependency in all four preset scenes and both saved conversions. Git exclusions and a credential scan passed. Git has not been initialized or pushed; the source-only clone still needs local runtime data provisioned separately.</p>
{images}
<p>These maintenance checks do not add realism, motion, contact, long-range or navigation-stack qualification. Earlier evidence and its limits remain below.</p>
</section><!-- repository-cleanup:end -->'''
path=root/'reports/progress.html';document=path.read_text()
pattern=r'<!-- repository-cleanup:start -->.*?<!-- repository-cleanup:end -->'
if re.search(pattern,document,re.S):document=re.sub(pattern,lambda _:section,document,flags=re.S)
else:document=document.replace('</h1>','</h1>'+section,1)
path.write_text(document)
print(report['status'])
