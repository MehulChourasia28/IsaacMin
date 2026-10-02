// Sparse topology-preserving conversion. No automatic gap filling/closing.
#include <openvdb/openvdb.h>
#include <openvdb/tools/TopologyToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv) {
 try {
  if(argc!=4) throw std::runtime_error("usage: openvdb_worker occupied_xyz_i32.bin output_prefix voxel_size_m");
  openvdb::initialize();
  const double spacing=std::stod(argv[3]);
  if(!(spacing>=0.02 && spacing<=1.0)) throw std::runtime_error("voxel_size outside bounds");
  auto occupancy=openvdb::BoolGrid::create(false);
  occupancy->setTransform(openvdb::math::Transform::createLinearTransform(spacing));
  auto accessor=occupancy->getAccessor();
  std::ifstream input(argv[1],std::ios::binary);
  int32_t p[3]; size_t count=0;
  while(input.read(reinterpret_cast<char*>(p),sizeof(p))) {
   accessor.setValue(openvdb::Coord(p[0],p[1],p[2]),true); ++count;
  }
  if(!input.eof()||input.gcount()!=0||count==0) throw std::runtime_error("invalid coordinates");
  auto sdf=openvdb::tools::topologyToLevelSet(*occupancy,3,0,0,0);
  sdf->setName("terrain_sdf");
  auto sign=sdf->getConstAccessor(); size_t failures=0;
  for(auto it=occupancy->cbeginValueOn();it;++it) if(it.isVoxelValue()&&sign.getValue(it.getCoord())>=0)++failures;
  if(failures) throw std::runtime_error("inside signs not preserved");
  std::string prefix(argv[2]);
  openvdb::io::File file(prefix+".vdb"); file.write({sdf}); file.close();
  std::vector<openvdb::Vec3s> vertices;
  std::vector<openvdb::Vec3I> triangles;
  std::vector<openvdb::Vec4I> quads;
  openvdb::tools::volumeToMesh(*sdf,vertices,triangles,quads,0.0,0.0);
  std::ofstream obj(prefix+".obj");
  for(auto& v:vertices)obj<<"v "<<v.x()<<' '<<v.y()<<' '<<v.z()<<'\n';
  // OpenVDB VolumeToMesh returns clockwise faces. OBJ/Blender expect outward
  // counter-clockwise winding; verified independently by signed-volume rays.
  for(auto& v:triangles)obj<<"f "<<v.x()+1<<' '<<v.z()+1<<' '<<v.y()+1<<'\n';
  for(auto& v:quads)obj<<"f "<<v.x()+1<<' '<<v.w()+1<<' '<<v.z()+1<<' '<<v.y()+1<<'\n';
  if(!obj||vertices.empty())throw std::runtime_error("empty/invalid mesh");
  std::cout<<"{\"status\":\"success\",\"backend\":\"OpenVDB\",\"version\":\""<<OPENVDB_LIBRARY_VERSION_STRING<<"\",\"occupied_voxels\":"<<count<<",\"vertices\":"<<vertices.size()<<",\"inside_sign_failures\":"<<failures<<",\"closing_steps\":0,\"smoothing_steps\":0}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
