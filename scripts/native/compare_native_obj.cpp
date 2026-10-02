// Independent streaming comparison of authoritative OBJ and native arrays.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
struct Mapped{int fd;size_t size;void*data;Mapped(const char*p){fd=open(p,O_RDONLY);if(fd<0)throw std::runtime_error("input_open");struct stat s;if(fstat(fd,&s))throw std::runtime_error("input_stat");size=s.st_size;data=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);if(data==MAP_FAILED)throw std::runtime_error("input_mmap");}~Mapped(){munmap(data,size);close(fd);}};
void skip(const char*&p){while(*p==' '||*p=='\t'||*p=='\r')++p;}
int main(int argc,char**argv){
 if(argc!=9){std::cerr<<"mesh.obj vertices.npy offset count triangles.npy offset count output.json\n";return 2;}
 std::size_t nv=0,nf=0,vc=0,fc=0,position_mismatch=0,index_mismatch=0;double maximum_error=0;std::string error;bool complete=false;
 try{Mapped vertices(argv[2]),faces(argv[5]);auto vo=std::stoull(argv[3]);nv=std::stoull(argv[4]);auto fo=std::stoull(argv[6]);nf=std::stoull(argv[7]);
  if(!nv||!nf||nv>(vertices.size-std::min(vertices.size,size_t(vo)))/12||nf>(faces.size-std::min(faces.size,size_t(fo)))/12)throw std::runtime_error("array_bounds");
  const float*v=reinterpret_cast<const float*>(static_cast<const char*>(vertices.data)+vo);const std::int32_t*f=reinterpret_cast<const std::int32_t*>(static_cast<const char*>(faces.data)+fo);
  std::ifstream obj(argv[1]);if(!obj)throw std::runtime_error("obj_open");std::string line;
  while(std::getline(obj,line)){const char*p=line.c_str();skip(p);
   if(p[0]=='v'&&(p[1]==' '||p[1]=='\t')){++p;if(vc>=nv)throw std::runtime_error("extra_obj_vertices");bool mismatch=false;
    for(int i=0;i<3;++i){skip(p);char*end;double value=std::strtod(p,&end);if(end==p||!std::isfinite(value)||!std::isfinite(v[vc*3+i]))throw std::runtime_error("invalid_vertex_coordinate");p=end;double difference=std::abs(value-double(v[vc*3+i]));maximum_error=std::max(maximum_error,difference);mismatch|=difference!=0.;}
    skip(p);if(*p&&*p!='#')throw std::runtime_error("unsupported_vertex_fields");position_mismatch+=mismatch;++vc;
   }else if(p[0]=='f'&&(p[1]==' '||p[1]=='\t')){++p;if(fc>=nf)throw std::runtime_error("extra_obj_faces");bool mismatch=false;
    for(int i=0;i<3;++i){skip(p);char*end;long long value=std::strtoll(p,&end,10);if(end==p||!value)throw std::runtime_error("invalid_face_index");p=end;long long index=value>0?value-1:static_cast<long long>(vc)+value;if(index<0||index>=static_cast<long long>(nv))throw std::runtime_error("face_index_out_of_range");mismatch|=index!=f[fc*3+i];if(*p=='/')while(*p&&*p!=' '&&*p!='\t'&&*p!='\r')++p;else if(*p&&*p!=' '&&*p!='\t'&&*p!='\r'&&*p!='#')throw std::runtime_error("invalid_face_token");}
    skip(p);if(*p&&*p!='#')throw std::runtime_error("nontriangular_face");index_mismatch+=mismatch;++fc;
   }
  }if(!obj.eof())throw std::runtime_error("obj_read_failure");complete=vc==nv&&fc==nf;if(!complete)error="missing_obj_rows";
 }catch(const std::exception&e){error=e.what();}
 std::ofstream out(argv[8]);out.precision(17);out<<"{\"kind\":\"NativeOBJArrayExactComparison\",\"complete\":"<<(complete?"true":"false")<<",\"vertices_compared\":"<<vc<<",\"triangles_compared\":"<<fc<<",\"coordinate_mismatch_vertices\":"<<position_mismatch<<",\"index_mismatch_triangles\":"<<index_mismatch<<",\"maximum_coordinate_error_m\":"<<maximum_error<<",\"error\":\""<<error<<"\"}\n";
 return complete&&!position_mismatch&&!index_mismatch?0:3;
}
