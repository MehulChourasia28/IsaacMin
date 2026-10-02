"""Algorithmic reuse checks only; these fixtures do not qualify Isaac visuals."""
import gzip
import numpy as np
import pytest
from nbtlib import Compound, Int, IntArray
from test_source import nbt_bytes
from isaacmin.io import atomic_json,sha256_file
from isaacmin.outdoor_pipeline import source_center
from isaacmin.assembly.surface_population import populate_surface
from isaacmin.source.snapshot import snapshot_world


@pytest.mark.parametrize('modern',[False,True])
def test_new_save_uses_its_own_level_spawn(tmp_path,modern):
    configured=tmp_path/'configured';other=tmp_path/'other';other.mkdir()
    atomic_json(tmp_path/'state/resolved_project.json',dict(source_world=str(configured),centre_minecraft_xz=[-1000,600]))
    spawn={'spawn':Compound({'pos':IntArray([1700023,75,-2300019])})} if modern else {
        'SpawnX':Int(1700023),'SpawnY':Int(75),'SpawnZ':Int(-2300019)}
    (other/'level.dat').write_bytes(gzip.compress(nbt_bytes({'Data':Compound(spawn)})))
    assert source_center(tmp_path,other)==((1700023.,-2300019.),'level.dat_spawn')
    assert source_center(tmp_path,configured)==((-1000.,600.),'saved_user_preference_for_this_world')
    assert source_center(tmp_path,other,(31.5,48.5))==((31.5,48.5),'command_line')


def test_ground_cover_does_not_reshuffle_when_local_origin_moves(tmp_path):
    # Flat geometry is an algorithm fixture. Both runs see identical source
    # columns; only their local USD coordinate origins differ.
    assets=[]
    for aid,name in [('grass_medium_01','grass_small_a'),('grass_medium_02','grass_a'),
                     ('dandelion_01','dandelion'),('celandine_01','celandine'),('fern_02','fern')]:
        assets.append(dict(asset_id=aid,output_sha256=aid,objects=[dict(object_name=name,
            dimensions_m=[.1,.1,.2],contact_anchors_local_m=[[0.,0.,0.]])]))
    atomic_json(tmp_path/'state/normalized_assets.json',dict(assets=assets))
    objects=tmp_path/'objects';objects.mkdir()
    np.savez(objects/'canopy.npz',cover_fraction=np.zeros((32,32)))
    fields=dict(height=np.zeros((32,32)),validity=np.ones((32,32),bool),min_xz=np.array([-16,-16]),
        water_validity=np.zeros((32,32),bool),water_height=np.zeros((32,32)),
        block_names=np.array(['minecraft:grass_block']),substrate_id=np.zeros((32,32),int),
        biome=np.full((32,32),'minecraft:plains'))
    results=[]
    for i,origin in enumerate(([0.,0.,0.],[3.125,0.,-5.875])):
        terrain=tmp_path/('terrain'+str(i));terrain.mkdir()
        np.save(terrain/'height.npy',np.zeros((65,65),np.float32))
        np.save(terrain/'vertices.npy',np.zeros((65*65,3),np.float32))
        np.savez(terrain/'material_fields.npz',**fields)
        atomic_json(terrain/'terrain.json',dict(recipe={'spacing_m':.125},bounds_source_xz=[-4.,-4.,4.,4.],origin_xyz=origin))
        out=tmp_path/('population'+str(i))
        populate_surface(tmp_path,terrain,objects,out)
        data=dict(np.load(out/'placements.npz'))
        data['source_positions']=data['positions'].astype(float)+[origin[0],-origin[2],origin[1]]
        results.append(data)
    a,b=results
    assert len(a['positions'])>100
    assert np.array_equal(a['world_cells'],b['world_cells'])
    assert np.array_equal(a['prototype_indices'],b['prototype_indices'])
    assert np.array_equal(a['scales'],b['scales'])
    assert np.array_equal(a['orientations_wxyz'],b['orientations_wxyz'])
    assert np.max(np.abs(a['source_positions']-b['source_positions']))<1e-6


def test_archive_spawn_is_read_from_immutable_extraction(tmp_path):
    import zipfile
    archive=tmp_path/'new_save.zip'
    level=gzip.compress(nbt_bytes({'Data':Compound({'SpawnX':Int(915),'SpawnY':Int(78),'SpawnZ':Int(-623)})}))
    with zipfile.ZipFile(archive,'w') as stream:
        stream.writestr('World/level.dat',level)
        stream.writestr('World/region/r.0.0.mca',b'fixture region bytes; no native terrain claim')
    before=archive.read_bytes()
    snapshot=snapshot_world(archive,tmp_path/'snapshots')
    assert source_center(tmp_path,archive,metadata_world=snapshot['snapshot_path'])==((915.,-623.),'level.dat_spawn')
    assert archive.read_bytes()==before


def test_replacing_a_map_in_the_same_folder_does_not_reuse_its_center(tmp_path):
    world=tmp_path/'save';world.mkdir()
    def level(x,z):
        return gzip.compress(nbt_bytes({'Data':Compound({'SpawnX':Int(x),'SpawnY':Int(70),'SpawnZ':Int(z)})}))
    (world/'level.dat').write_bytes(level(10,20))
    atomic_json(tmp_path/'state/resolved_project.json',dict(source_world=str(world),
        centre_minecraft_xz=[-1065.38,688.07],preferred_source_metadata_sha256=sha256_file(world/'level.dat')))
    assert source_center(tmp_path,world)[0]==(-1065.38,688.07)
    (world/'level.dat').write_bytes(level(420,-800))
    assert source_center(tmp_path,world)==((420.,-800.),'level.dat_spawn')


def test_registered_centers_are_bound_to_each_actual_save_identity(tmp_path):
    worlds=[]
    for i in range(4):
        world=tmp_path/('map'+str(i));world.mkdir()
        (world/'level.dat').write_bytes(gzip.compress(nbt_bytes({'Data':Compound({
            'SpawnX':Int(100+i),'SpawnY':Int(70),'SpawnZ':Int(-100-i)})})))
        worlds.append(dict(id='map'+str(i),source=world.name,
            center_minecraft_xz=[20.25+i*1000,-35.5-i*1000],
            level_dat_sha256=sha256_file(world/'level.dat')))
    atomic_json(tmp_path/'state/source_worlds.json',dict(worlds=worlds))
    for entry in worlds:
        world=tmp_path/entry['source']
        assert source_center(tmp_path,world)==(tuple(entry['center_minecraft_xz']),
            'registered_user_preference:'+entry['id'])
    replacement=tmp_path/worlds[2]['source']/'level.dat'
    replacement.write_bytes(gzip.compress(nbt_bytes({'Data':Compound({
        'SpawnX':Int(9),'SpawnY':Int(70),'SpawnZ':Int(-11)})})))
    assert source_center(tmp_path,replacement.parent)==((9.,-11.),'level.dat_spawn')


def test_nearly_aligned_negative_center_keeps_the_complete_square(tmp_path):
    # Four complete chunks occupy [-16,16)^2. The old floor selected a missing
    # western column for center x=-0.6, despite that center being inside it.
    from test_source import make_region,chunk_nbt
    from isaacmin.source.macro import extract_macro_surface
    from isaacmin.source.semantics import classify
    world=tmp_path/'save';world.mkdir();(world/'level.dat').write_bytes(b'hashed fixture metadata')
    for x in (-1,0):
        for z in (-1,0):
            make_region(world/f'region/r.{x//32}.{z//32}.mca',chunk_nbt(x,z),x,z)
    result=extract_macro_surface(world,tmp_path/'macro',(-.6,1.2),extent=32,spacing=8)
    assert result['scope']['bounds_blocks_xz']==[-16,-16,16,16]
    assert result['scope']['full_chunks']==4
    assert result['surface']['invalid_samples']==0
    assert classify('minecraft:mangrove_propagule')=='vegetation'
    assert classify('minecraft:sea_pickle')=='vegetation'
    assert classify('custom:sea_pickle')=='unknown'
