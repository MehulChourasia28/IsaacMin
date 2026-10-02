// Read-only bounded CGAL segment queries against every original float32 face.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/intersections.h>
#include <boost/variant/get.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
using K=CGAL::Exact_predicates_inexact_constructions_kernel;
using Point=K::Point_3;using Triangle=K::Triangle_3;using Segment=K::Segment_3;
using Iterator=std::vector<Triangle>::iterator;
using Primitive=CGAL::AABB_triangle_primitive<K,Iterator>;
using Tree=CGAL::AABB_tree<CGAL::AABB_traits<K,Primitive>>;
using Intersection=Tree::Intersection_and_primitive_id<Segment>::Type;
using Bounds=std::array<double,6>;
struct Mapped{int fd;std::size_t size;void*data;Mapped(const char*p){fd=open(p,O_RDONLY);if(fd<0)throw std::runtime_error("input_open");struct stat s;if(fstat(fd,&s))throw std::runtime_error("input_stat");size=s.st_size;data=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);if(data==MAP_FAILED)throw std::runtime_error("input_mmap");}~Mapped(){munmap(data,size);close(fd);}};
struct Box{std::array<float,3>lo,hi;};
struct Record{std::uint64_t query,triangle,kind;double first[3],second[3],normal[3];};
static_assert(sizeof(Record)==96,"Native output layout changed");
struct Runner{
 const float*v;const std::int32_t*f;std::vector<Box>boxes;std::vector<Segment>queries;
 std::size_t maximum_leaf;std::uint64_t max_hits,hits=0,leaves=0,face_visits=0,indexed_visits=0,query_visits=0,max_observed=0;
 std::ofstream output,progress;
 Point point(std::size_t id){return Point(double(v[3*id]),double(v[3*id+1]),double(v[3*id+2]));}
 void visit(std::vector<std::size_t>ids,Bounds cell,int depth){
  if(ids.empty())return;
  if(ids.size()>maximum_leaf){
   if(depth>=64)throw std::runtime_error("spatial_partition_depth_exceeded");
   std::array<int,3>axes{0,1,2};std::sort(axes.begin(),axes.end(),[&](int a,int b){return cell[a+3]-cell[a]>cell[b+3]-cell[b];});
   for(int axis:axes){double m=cell[axis]+(cell[axis+3]-cell[axis])/2;if(m==cell[axis]||m==cell[axis+3])continue;
    std::vector<std::size_t>left,right;left.reserve(ids.size()/2);right.reserve(ids.size()/2);
    for(auto id:ids){if(boxes[id].lo[axis]<=m)left.push_back(id);if(boxes[id].hi[axis]>=m)right.push_back(id);}
    if(left.size()==ids.size()&&right.size()==ids.size())continue;
    Bounds a=cell,b=cell;a[axis+3]=m;b[axis]=m;std::vector<std::size_t>().swap(ids);visit(std::move(left),a,depth+1);visit(std::move(right),b,depth+1);return;
   }throw std::runtime_error("spatial_cell_cannot_fit_without_omitting_faces");
  }
  K::Iso_cuboid_3 cube(cell[0],cell[1],cell[2],cell[3],cell[4],cell[5]);std::vector<std::size_t>selected;
  for(std::size_t q=0;q<queries.size();++q)if(CGAL::do_intersect(queries[q],cube))selected.push_back(q);
  face_visits+=ids.size();++leaves;max_observed=std::max<std::uint64_t>(max_observed,ids.size());
  if(!selected.empty()){
   std::vector<Triangle>triangles;triangles.reserve(ids.size());
   for(auto id:ids){triangles.emplace_back(point(f[3*id]),point(f[3*id+1]),point(f[3*id+2]));if(triangles.back().is_degenerate())throw std::runtime_error("degenerate_original_triangle");}
   Tree tree(triangles.begin(),triangles.end());tree.build();indexed_visits+=ids.size();query_visits+=selected.size();
   for(auto q:selected){std::vector<Intersection>found;tree.all_intersections(queries[q],std::back_inserter(found));
    for(const auto&hit:found){
     if(hits>=max_hits)throw std::runtime_error("intersection_output_capacity_exceeded");
     std::size_t local=hit.second-triangles.begin();Record record{};record.query=q;record.triangle=ids[local];
     Point a,b;if(const Point*p=boost::get<Point>(&hit.first)){a=b=*p;record.kind=0;}else if(const Segment*s=boost::get<Segment>(&hit.first)){a=s->source();b=s->target();record.kind=1;}else throw std::runtime_error("unexpected_intersection_object");
     auto normal=CGAL::cross_product(triangles[local][1]-triangles[local][0],triangles[local][2]-triangles[local][0]);
     for(int axis=0;axis<3;++axis){record.first[axis]=CGAL::to_double(a[axis]);record.second[axis]=CGAL::to_double(b[axis]);record.normal[axis]=CGAL::to_double(normal[axis]);}
     output.write(reinterpret_cast<const char*>(&record),sizeof(record));if(!output)throw std::runtime_error("intersection_output_write");++hits;
    }
   }
  }
  progress<<"{\"leaf\":"<<leaves<<",\"faces\":"<<ids.size()<<",\"queries\":"<<selected.size()<<",\"cumulative_raw_hits\":"<<hits<<"}\n";progress.flush();
 }
};
int main(int argc,char**argv){
 if(argc!=13){std::cerr<<"vertices.npy offset count triangles.npy offset count segments.npy offset count outdir max_leaf_faces max_hits\n";return 2;}
 auto started=std::chrono::steady_clock::now();Runner runner;std::string error;bool complete=false;std::size_t nv=0,nf=0,nq=0;std::string directory=argv[10];runner.output.open(directory+"/segment_hits.bin",std::ios::binary);runner.progress.open(directory+"/progress.jsonl");
 try{
  Mapped vertices(argv[1]),faces(argv[4]),queries(argv[7]);auto vo=std::stoull(argv[2]);nv=std::stoull(argv[3]);auto fo=std::stoull(argv[5]);nf=std::stoull(argv[6]);auto qo=std::stoull(argv[8]);nq=std::stoull(argv[9]);
  if(!nv||!nf||!nq||nv>(vertices.size-std::min(vertices.size,std::size_t(vo)))/12||nf>(faces.size-std::min(faces.size,std::size_t(fo)))/12||nq>(queries.size-std::min(queries.size,std::size_t(qo)))/48)throw std::runtime_error("invalid_array_bounds");
  runner.v=reinterpret_cast<const float*>(static_cast<const char*>(vertices.data)+vo);runner.f=reinterpret_cast<const std::int32_t*>(static_cast<const char*>(faces.data)+fo);auto q=reinterpret_cast<const double*>(static_cast<const char*>(queries.data)+qo);
  runner.maximum_leaf=std::stoull(argv[11]);runner.max_hits=std::stoull(argv[12]);if(!runner.maximum_leaf||!runner.max_hits)throw std::runtime_error("invalid_limits");
  Bounds global{INFINITY,INFINITY,INFINITY,-INFINITY,-INFINITY,-INFINITY};
  for(std::size_t i=0;i<nv;++i)for(int a=0;a<3;++a){double value=runner.v[3*i+a];if(!std::isfinite(value))throw std::runtime_error("nonfinite_vertex");global[a]=std::min(global[a],value);global[a+3]=std::max(global[a+3],value);}
  runner.boxes.resize(nf);std::vector<std::size_t>ids(nf);
  for(std::size_t i=0;i<nf;++i){ids[i]=i;Box box{{INFINITY,INFINITY,INFINITY},{-INFINITY,-INFINITY,-INFINITY}};for(int c=0;c<3;++c){auto id=runner.f[3*i+c];if(id<0||std::size_t(id)>=nv)throw std::runtime_error("invalid_face_index");for(int a=0;a<3;++a){box.lo[a]=std::min(box.lo[a],runner.v[id*3+a]);box.hi[a]=std::max(box.hi[a],runner.v[id*3+a]);}}runner.boxes[i]=box;}
  runner.queries.reserve(nq);for(std::size_t i=0;i<nq;++i){for(int a=0;a<6;++a)if(!std::isfinite(q[6*i+a]))throw std::runtime_error("nonfinite_query");runner.queries.emplace_back(Point(q[6*i],q[6*i+1],q[6*i+2]),Point(q[6*i+3],q[6*i+4],q[6*i+5]));if(runner.queries.back().is_degenerate())throw std::runtime_error("degenerate_query");}
  runner.visit(std::move(ids),global,0);complete=true;
 }catch(const std::exception&e){error=e.what();}catch(...){error="unclassified_native_exception";}
 runner.output.close();runner.progress.close();struct rusage usage;getrusage(RUSAGE_SELF,&usage);double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();std::ofstream report(directory+"/native_result.json");
 report<<"{\"schema_version\":1,\"kind\":\"NativeCGALSegmentIntersections\",\"complete\":"<<(complete?"true":"false")<<",\"vertices\":"<<nv<<",\"original_triangles\":"<<nf<<",\"queries\":"<<nq<<",\"raw_intersections\":"<<runner.hits<<",\"completed_leaves\":"<<runner.leaves<<",\"leaf_face_visits\":"<<runner.face_visits<<",\"indexed_face_visits\":"<<runner.indexed_visits<<",\"query_leaf_visits\":"<<runner.query_visits<<",\"maximum_leaf_faces\":"<<runner.maximum_leaf<<",\"maximum_observed_leaf_faces\":"<<runner.max_observed<<",\"peak_rss_kib\":"<<usage.ru_maxrss<<",\"elapsed_seconds\":"<<elapsed<<",\"error\":\""<<error<<"\",\"method\":\"CGAL exact predicates with double-precision intersection constructions; all original faces retained, shared spatial-cell query duplicates emitted for identity deduplication; coplanar segment intersections retained explicitly\"}\n";
 return complete?0:3;
}
