"""Licence snapshots, visible credits and evidence-qualified coverage matrix."""
import hashlib
import json
from pathlib import Path

from .network import PublicClient, ServiceError, atomic_json, utcnow
from .references import build_reference_board


def finalize_catalogue(workspace: Path, *, fetch_licences=True):
    workspace=Path(workspace).resolve()
    path=workspace/'state/asset_catalogue.json'
    catalogue=json.loads(path.read_text())
    licence_records={}
    folder=workspace/'evidence/services/licences';folder.mkdir(parents=True,exist_ok=True)
    sources={'poly_haven':'https://polyhaven.com/license','ambientcg':'https://docs.ambientcg.com/license/',
             'poly_haven_api_terms':'https://raw.githubusercontent.com/Poly-Haven/Public-API/master/ToS.md'}
    manifest=folder/'manifest.json'
    if manifest.exists():licence_records=json.loads(manifest.read_text())
    if fetch_licences:
        client=PublicClient(workspace/'cache/assets/licences',('polyhaven.com','ambientcg.com','raw.githubusercontent.com'))
        for provider,url in sources.items():
            r=client._request('GET',url)
            if r.status_code!=200 or len(r.content)>2*1024**2:
                raise ServiceError('licence_unavailable','Required licence/API terms snapshot unavailable')
            target=folder/(provider+'.txt');target.write_bytes(r.content)
            licence_records[provider]={'url':url,'path':str(target),'sha256':hashlib.sha256(r.content).hexdigest(),'retrieved_at_utc':utcnow()}
        atomic_json(manifest,licence_records)
    for a in catalogue['assets']:
        if a['provider'] in licence_records:
            a['licence_text_sha256']=licence_records[a['provider']]['sha256']
            a['licence_snapshot_path']=licence_records[a['provider']]['path']
        if a['asset_id'].startswith('pine_'):
            a['scene_use']='diagnostic_only; incompatible with selected temperate deciduous/plains source scope'
    atomic_json(path,catalogue)
    credits=['# Asset credits','','Assets from Poly Haven. Additional assets from ambientCG.','',
             'Downloaded candidates and procedural assets remain unqualified until actual Isaac tests pass.','']
    for a in catalogue['assets']:
        credits.append(f"- [{a['asset_id']}]({a['source_page']}) — {a['provider']}; authors: {', '.join(a['authors'])}; [CC0-1.0]({a['licence_url']}).")
    credits+=['','Procedural white-birch and oak candidates use Blender Sapling Tree Gen 0.3.5 '
              '(GPL-2.0-or-later generator). Generator source hashes and original notices are recorded in '
              'assets/procedural/{white_birch,oak}/generation.json. White-birch maps are explicitly procedural. '
              'Oak bark and leaves use the unchanged CC0 ambientCG Bark012 and LeafSet016 maps credited above; '
              'leaf outlines are traced from the actual opacity atlas. Generated tree geometry is not '
              'represented as a photographic scan. Neither canopy candidate is Isaac-qualified.']
    (workspace/'ASSET_CREDITS.md').write_text('\n'.join(credits)+'\n')
    references=[]
    for a in catalogue['assets']:
        if a['kind']=='hdris' and not a['asset_id'].endswith('_puresky'):
            references.append(build_reference_board(workspace,a))
    atomic_json(workspace/'state/reference_catalogue.json',{'references':references,'synthetic_reference_count':0})
    coverage={'schema_version':1,'created_at_utc':utcnow(),'status':'incomplete_unqualified',
              'interpreted_ecology':'late spring temperate meadow with deciduous oak/birch woodland; inferred season',
              'source_biomes':['minecraft:plains','minecraft:forest','minecraft:old_growth_birch_forest'],
              'guilds':{'meadow_grasses':{'candidates':['grass_medium_01','grass_medium_02'],'geometric_variants':27,'species_identification':'unspecified provider grass; botanically limited'},
                        'meadow_forbs':{'candidates':['dandelion_01'],'geometric_variants':5},
                        'spring_woodland_forbs':{'candidates':['celandine_01'],'geometric_variants':5},
                        'shade_understory':{'candidates':['fern_02'],'geometric_variants':4},
                        'deadwood':{'candidates':['dead_tree_trunk'],'geometric_variants':1},
                        'mossy_rocks':{'candidates':['rock_moss_set_01'],'geometric_variants':6,'lithology':'not identified'},
                        'birch_canopy':{'candidates':['procedural_white_birch'],'geometric_variants':3,'native_generated':True,'botanical_qualification':'not_run'},
                        'oak_canopy':{'candidates':[],'available_maps':['LeafSet016','Bark012'],'gap':'native oak branching/assembly and qualification pending'},
                        'clover':{'candidates':[],'gap':'no suitable mesh found in both queried public catalogues'},
                        'litter':{'candidates':['forest_ground_04','Ground106'],'gap':'material-only litter cannot replace near-field individual geometry'}},
              'renderer_qualification':'not_run','no_conifer_substitution':True,
              'reference_limits':'Different geographic locations, capture heights undocumented; panorama directions recorded; no metric scene ground truth'}
    for guild,species in [('birch_canopy','white_birch'),('oak_canopy','oak')]:
        generated=workspace/'assets/procedural'/species/'generation.json'
        if generated.exists():
            record=json.loads(generated.read_text())
            coverage['guilds'][guild].update(candidates=[record['asset_id']],native_generated=True,
                                             geometric_variants=len(record['variants']),
                                             botanical_qualification='not_run',
                                             appearance_defect='Sparse or clustered crown in actual Blender diagnostic; Isaac check not_run')
            coverage['guilds'][guild].pop('gap',None)
    atomic_json(workspace/'state/biome_asset_coverage.json',coverage)
    return coverage
