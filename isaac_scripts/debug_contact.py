from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
try:
 import json,numpy as np,omni.usd,omni.physx
 from pathlib import Path
 from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,UsdShade
 from isaacsim.core.api import World
 from isaacsim.core.api.objects import DynamicCuboid
 base=Path(__file__).resolve().parents[1]/'artifacts/bootstrap'
 omni.usd.get_context().open_stage(str(base/'blender/world.usda'))
 for _ in range(30):app.update()
 stage=omni.usd.get_context().get_stage()
 world=World(stage_units_in_meters=1,physics_dt=1/120,rendering_dt=1/120)
 world.get_physics_context().enable_gpu_dynamics(True)
 world.get_physics_context().set_broadphase_type('GPU')
 world.scene.add_default_ground_plane()
 terrain=stage.GetPrimAtPath('/World/Terrain_FinalGround/terrain')
 UsdPhysics.CollisionAPI.Apply(terrain).CreateCollisionEnabledAttr(True)
 UsdPhysics.CollisionAPI(terrain).CreateSimulationOwnerRel().SetTargets(['/physicsScene'])
 UsdPhysics.MeshCollisionAPI.Apply(terrain).CreateApproximationAttr('none')
 original_mesh=UsdGeom.Mesh(terrain)
 clone=UsdGeom.Mesh.Define(stage,'/World/CollisionClone')
 clone.CreatePointsAttr(original_mesh.GetPointsAttr().Get())
 counts=original_mesh.GetFaceVertexCountsAttr().Get();indices=original_mesh.GetFaceVertexIndicesAttr().Get()
 triangles=[];offset=0
 for count in counts:
  triangles.extend((indices[offset],indices[offset+j],indices[offset+j+1]) for j in range(1,count-1))
  offset+=count
 points=np.asarray(original_mesh.GetPointsAttr().Get())
 triangles=np.asarray(triangles,dtype=np.int32)
 centers=points[triangles].mean(axis=1)
 order=np.argsort((centers[:,0]-3.5)**2+(centers[:,1]+3.5)**2+(centers[:,2]-9)**2)
 triangles=triangles[order[:2]]
 unique,inverse=np.unique(triangles,return_inverse=True)
 clone.CreatePointsAttr(points[unique].tolist())
 triangles=inverse.reshape(-1,3)
 clone.CreateFaceVertexCountsAttr([3]*len(triangles))
 clone.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
 clone.CreateSubdivisionSchemeAttr('none')
 UsdGeom.Xformable(clone).AddTranslateOp().Set(Gf.Vec3d(30,0,0))
 UsdPhysics.CollisionAPI.Apply(clone.GetPrim()).CreateCollisionEnabledAttr(True)
 UsdPhysics.CollisionAPI(clone.GetPrim()).CreateSimulationOwnerRel().SetTargets(['/physicsScene'])
 UsdPhysics.MeshCollisionAPI.Apply(clone.GetPrim()).CreateApproximationAttr('none')
 mat=UsdShade.Material.Define(stage,'/World/PhysicsGround')
 physics_material=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
 physics_material.CreateStaticFrictionAttr(.8)
 physics_material.CreateDynamicFrictionAttr(.6)
 physics_material.CreateRestitutionAttr(0)
 for prim in [terrain,clone.GetPrim(),*terrain.GetChildren()]:
  UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,materialPurpose='physics')
 flat=UsdGeom.Mesh.Define(stage,'/World/FlatDiagnostic')
 flat.CreatePointsAttr([(-11,-4,5),(-9,-4,5),(-9,-2,5),(-11,-2,5)])
 flat.CreateFaceVertexCountsAttr([3,3]);flat.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
 flat.CreateSubdivisionSchemeAttr('none')
 UsdPhysics.CollisionAPI.Apply(flat.GetPrim()).CreateCollisionEnabledAttr(True)
 UsdPhysics.MeshCollisionAPI.Apply(flat.GetPrim()).CreateApproximationAttr('none')
 UsdShade.MaterialBindingAPI.Apply(flat.GetPrim()).Bind(mat,materialPurpose='physics')
 box=world.scene.add(DynamicCuboid('/World/Probe',position=np.array([3.5,-3.5,10]),
             scale=np.array([.2,.2,.2]),mass=1))
 box2=world.scene.add(DynamicCuboid('/World/Probe2',name='probe2',position=np.array([33.5,-3.5,10]),
             scale=np.array([.2,.2,.2]),mass=1))
 box3=world.scene.add(DynamicCuboid('/World/Probe3',name='probe3',position=np.array([-10,-3,6]),
             scale=np.array([.2,.2,.2]),mass=1))
 for path in ['/World/Probe','/World/Probe2']:
  UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).CreateSimulationOwnerRel().SetTargets(['/physicsScene'])
 world.reset()
 query=omni.physx.get_physx_scene_query_interface()
 logs=[]
 for i in range(240):
  world.step(render=True)
  if i%20==0:
   ray=query.raycast_closest((3.5,-3.5,12),(0,0,-1),100)
   entry={'step':i,'position':box.get_world_pose()[0].tolist(),
          'clone_position':box2.get_world_pose()[0].tolist(),
          'flat_position':box3.get_world_pose()[0].tolist(),'ray':str(ray)}
   logs.append(entry);print('CONTACT_DEBUG',entry,flush=True)
 (base/'contact_debug.json').write_text(json.dumps(logs,indent=2))
 stage.GetRootLayer().Export(str(base/'contact_debug.usda'))
except BaseException:
 import traceback;print(traceback.format_exc(),flush=True)
finally:app.close()
