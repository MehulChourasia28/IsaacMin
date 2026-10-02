import math
import numpy as np
import pytest

from isaacmin.assembly.ecology import RULES, ground_prototype, rule_violations, scatter_candidates


def fields(x,y):
    return {'valid':True,'biome':'minecraft:plains','substrate':'soil','slope_degrees':0,
            'moisture':.5,'canopy':.1,'water_depth_m':0,'route_distance_m':10}


def support(x,y,hint):
    return {'z':.1*x,'normal':[-.1,0,1],'surface_id':'final','mesh_sha256':'verified-final-mesh'}


def prototype():
    return {'object_name':'plant','contact_anchors_local_m':[[-.02,0,0],[.02,0,0],[0,.02,0]]}


def assets():
    return [{'asset_id':'grass_medium_01','objects':[prototype()],'output_blend':'candidate.blend','output_sha256':'abc','isaac_qualification':'not_run'}]


def test_no_unqualified_asset_enters_strict_placement():
    r=scatter_candidates([0,0,3,3],fields,support,assets())
    assert r['instances']==[]
    assert r['coverage_gaps']


def test_base_fits_sloped_final_ground_not_source_height():
    r,error=ground_prototype(prototype(),2,0,1,0,support)
    assert error is None
    assert r['contact_offset_max_m']<1e-6
    assert abs(r['world_transform'][2][3]-.2)<1e-6
    assert r['supporting_mesh_sha256']=='verified-final-mesh'


def test_contact_rejects_discontinuous_supporting_floor():
    def broken(x,y,hint):
        return {'z':0 if x<0 else .5,'normal':[0,0,1],'surface_id':'final','mesh_sha256':'verified-final-mesh'}
    r,error=ground_prototype(prototype(),0,0,1,0,broken)
    assert r is None and error=='root_contact_spread'


def test_plants_reject_water_rock_wrong_biome_and_missing_semantics():
    f=fields(0,0)
    f.update(water_depth_m=.01,substrate='rock',biome='minecraft:desert')
    failures=rule_violations(RULES[0],f)
    assert set(failures)>={'submerged','substrate_mismatch','biome_mismatch'}
    assert rule_violations(RULES[0],{})[0].startswith('missing_fields')


def test_world_cell_ownership_deterministic_across_tile_order():
    whole=scatter_candidates([-4,-4,4,4],fields,support,assets(),require_qualified=False)
    left=scatter_candidates([-4,-4,0,4],fields,support,assets(),require_qualified=False)
    right=scatter_candidates([0,-4,4,4],fields,support,assets(),require_qualified=False)
    w={x['instance_id']:x for x in whole['instances']}
    tiled={x['instance_id']:x for x in right['instances']+left['instances']}
    assert w and tiled==w
    assert all(x['contact_offset_max_m']<=.02 for x in w.values())


def test_empty_cells_do_not_suppress_rare_deadwood_guild():
    def woodland(x,y):
        return {**fields(x,y),'biome':'minecraft:forest','canopy':.6}
    logs=[{**assets()[0],'asset_id':'dead_tree_trunk'}]
    result=scatter_candidates([0,0,64,64],woodland,support,logs,require_qualified=False)
    assert 6 <= len(result['instances']) <= 35
    xy=np.array([[i['world_transform'][0][3],i['world_transform'][1][3]] for i in result['instances']])
    distances=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=-1)
    np.fill_diagonal(distances,np.inf)
    assert distances.min()>=2.5


def test_real_normalization_preserved_multiple_geometric_variants():
    from pathlib import Path
    import json
    import pytest
    path=Path(__file__).resolve().parents[1]/'state/normalized_assets.json'
    if not path.exists():pytest.skip('Native normalization has not run')
    records=json.loads(path.read_text())['assets']
    assert sum(len(a['objects']) for a in records)>=40
    assert all(abs(o['normalized_contact_z'])<1e-5 for a in records for o in a['objects'])
    assert all(a['isaac_qualification']=='not_run' for a in records)
def test_exact_route_clearance_rejects_jittered_crossing_of_flora_buffer():
    from isaacmin.assembly.ecology import PolylineClearance,rule_violations,RULES
    route=PolylineClearance([[0,0,0],[10,0,0]],1)
    assert route(5,1)==.5  # raster cell center would have passed
    assert route(5,.9)==pytest.approx(.4)  # actual jittered root violates0.45m
    field={'valid':True,'biome':'plains','substrate':'soil','slope_degrees':0,'moisture':.5,'canopy':.2,'water_depth_m':0,'route_distance_m':route(5,.9)}
    assert 'trail_clearance' in rule_violations(RULES[0],field)
    field['route_distance_m']=route(5,.96)
    assert not rule_violations(RULES[0],field)
