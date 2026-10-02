// Independent read-only CGAL triangle-soup validation. No mesh repair.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

using Kernel=CGAL::Exact_predicates_inexact_constructions_kernel;
using Point=Kernel::Point_3;
using Triangle=std::array<std::size_t,3>;
using Bounds=std::array<double,6>;
struct Mapped {
  int fd;std::size_t size;void* data;
  Mapped(const char* path){fd=open(path,O_RDONLY);if(fd<0)throw std::runtime_error("input_open");struct stat s;if(fstat(fd,&s))throw std::runtime_error("input_stat");size=s.st_size;data=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);if(data==MAP_FAILED)throw std::runtime_error("input_mmap");}
  ~Mapped(){munmap(data,size);close(fd);}
};
struct Box {std::array<float,3> lo,hi;};
struct State {
  const std::vector<Box>* boxes;const std::vector<std::size_t>* ids;Bounds cell,global;
  std::ofstream* stream;std::uint64_t* count;std::uint64_t maximum;
};
struct PairSink {
  std::shared_ptr<State> state;
  using iterator_category=std::output_iterator_tag;using value_type=void;using difference_type=std::ptrdiff_t;using pointer=void;using reference=void;
  PairSink& operator*(){return *this;}PairSink& operator++(){return *this;}PairSink operator++(int){return *this;}
  PairSink& operator=(const std::pair<std::size_t,std::size_t>& pair){
    std::size_t a=(*state->ids)[pair.first],b=(*state->ids)[pair.second];
    for(int axis=0;axis<3;++axis){double p=std::max((*state->boxes)[a].lo[axis],(*state->boxes)[b].lo[axis]);
      if(p<state->cell[axis] || (p>=state->cell[axis+3] && state->cell[axis+3]!=state->global[axis+3]))return *this;
    }
    if(*state->count>=state->maximum)throw std::runtime_error("pair_output_capacity_exceeded");
    std::array<std::uint64_t,2> record{std::min(a,b),std::max(a,b)};
    state->stream->write(reinterpret_cast<const char*>(record.data()),sizeof(record));if(!*state->stream)throw std::runtime_error("pair_output_write");++*state->count;return *this;
  }
};
struct Runner {
  std::vector<Point> points;std::vector<Box> boxes;const std::int32_t* face_data;
  Bounds global;std::ofstream output,progress;std::size_t maximum_leaf;std::uint64_t pairs=0,max_pairs,leaves=0,leaf_face_visits=0,max_observed_leaf=0;
  void visit(std::vector<std::size_t> ids,Bounds cell,int depth){
    if(ids.empty())return;
    if(ids.size()>maximum_leaf){
      if(depth>=64)throw std::runtime_error("spatial_partition_depth_exceeded");
      std::array<int,3> axes{0,1,2};std::sort(axes.begin(),axes.end(),[&](int a,int b){return cell[a+3]-cell[a]>cell[b+3]-cell[b];});
      for(int axis:axes){double middle=cell[axis]+(cell[axis+3]-cell[axis])/2;
        if(middle==cell[axis]||middle==cell[axis+3])continue;
        std::vector<std::size_t> left,right;left.reserve(ids.size()/2);right.reserve(ids.size()/2);
        for(auto id:ids){if(boxes[id].lo[axis]<=middle)left.push_back(id);if(boxes[id].hi[axis]>=middle)right.push_back(id);}
        if(left.size()==ids.size()&&right.size()==ids.size())continue;
        Bounds a=cell,b=cell;a[axis+3]=middle;b[axis]=middle;
        std::vector<std::size_t>().swap(ids);visit(std::move(left),a,depth+1);visit(std::move(right),b,depth+1);return;
      }
      throw std::runtime_error("spatial_cell_cannot_fit_without_omitting_faces");
    }
    std::vector<Triangle> triangles;triangles.reserve(ids.size());
    for(auto id:ids)triangles.push_back({std::size_t(face_data[id*3]),std::size_t(face_data[id*3+1]),std::size_t(face_data[id*3+2])});
    auto state=std::make_shared<State>(State{&boxes,&ids,cell,global,&output,&pairs,max_pairs});
    CGAL::Polygon_mesh_processing::triangle_soup_self_intersections<CGAL::Sequential_tag>(points,triangles,PairSink{state});
    ++leaves;leaf_face_visits+=ids.size();max_observed_leaf=std::max<std::uint64_t>(max_observed_leaf,ids.size());
    progress<<"{\"leaf\":"<<leaves<<",\"original_faces\":"<<ids.size()<<",\"cumulative_intersection_pairs\":"<<pairs<<"}\n";progress.flush();
  }
};
int main(int argc,char**argv){
  if(argc!=10){std::cerr<<"vertices.npy vertex_offset vertex_count triangles.npy face_offset face_count output_dir maximum_leaf_faces maximum_pairs\n";return 2;}
  auto started=std::chrono::steady_clock::now();Runner runner;std::string error;bool complete=false;std::size_t nv=0,nf=0;
  std::string directory=argv[7];runner.output.open(directory+"/intersection_pairs.u64",std::ios::binary);runner.progress.open(directory+"/progress.jsonl");
  try {
    Mapped vertices(argv[1]),faces(argv[4]);std::size_t vo=std::stoull(argv[2]);nv=std::stoull(argv[3]);std::size_t fo=std::stoull(argv[5]);nf=std::stoull(argv[6]);
    if(!nv||!nf||nv>(vertices.size-std::min(vertices.size,vo))/12||nf>(faces.size-std::min(faces.size,fo))/12)throw std::runtime_error("invalid_array_bounds");
    auto v=reinterpret_cast<const float*>(static_cast<const char*>(vertices.data)+vo);runner.face_data=reinterpret_cast<const std::int32_t*>(static_cast<const char*>(faces.data)+fo);
    runner.maximum_leaf=std::stoull(argv[8]);runner.max_pairs=std::stoull(argv[9]);if(!runner.maximum_leaf||!runner.max_pairs)throw std::runtime_error("invalid_limits");
    runner.points.reserve(nv);runner.global={INFINITY,INFINITY,INFINITY,-INFINITY,-INFINITY,-INFINITY};
    for(std::size_t i=0;i<nv;++i){for(int a=0;a<3;++a){if(!std::isfinite(v[3*i+a]))throw std::runtime_error("nonfinite_vertex");runner.global[a]=std::min(runner.global[a],double(v[3*i+a]));runner.global[a+3]=std::max(runner.global[a+3],double(v[3*i+a]));}runner.points.emplace_back(double(v[3*i]),double(v[3*i+1]),double(v[3*i+2]));}
    runner.boxes.resize(nf);std::vector<std::size_t> ids(nf);
    for(std::size_t i=0;i<nf;++i){ids[i]=i;Box box{{INFINITY,INFINITY,INFINITY},{-INFINITY,-INFINITY,-INFINITY}};
      for(int c=0;c<3;++c){auto id=runner.face_data[i*3+c];if(id<0||std::size_t(id)>=nv)throw std::runtime_error("invalid_face_index");for(int a=0;a<3;++a){box.lo[a]=std::min(box.lo[a],v[id*3+a]);box.hi[a]=std::max(box.hi[a],v[id*3+a]);}}runner.boxes[i]=box;}
    runner.visit(std::move(ids),runner.global,0);complete=true;
  }catch(const std::exception& e){error=e.what();}catch(...){error="unclassified_native_exception";}
  runner.output.close();runner.progress.close();struct rusage usage;getrusage(RUSAGE_SELF,&usage);
  double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  std::ofstream report(directory+"/native_result.json");
  report<<"{\"schema_version\":1,\"kind\":\"NativeCGALTriangleSoupIntersections\",\"status\":\""<<(complete?(runner.pairs?"fail":"pass"):"incomplete")<<"\",\"complete\":"<<(complete?"true":"false")<<",\"vertices\":"<<nv<<",\"original_triangles\":"<<nf<<",\"intersection_pairs\":"<<runner.pairs<<",\"completed_leaves\":"<<runner.leaves<<",\"leaf_face_visits\":"<<runner.leaf_face_visits<<",\"maximum_leaf_faces\":"<<runner.maximum_leaf<<",\"maximum_observed_leaf_faces\":"<<runner.max_observed_leaf<<",\"peak_rss_kib\":"<<usage.ru_maxrss<<",\"elapsed_seconds\":"<<elapsed<<",\"error\":\""<<error<<"\",\"kernel\":\"CGAL Exact_predicates_inexact_constructions_kernel; original float32 coordinates represented exactly as doubles, no geometry construction or repair\",\"partition_ownership\":\"Half-open spatial cells; each pair belongs to the cell containing the componentwise maximum of its two original AABB minima\"}\n";
  return complete?0:3;
}
