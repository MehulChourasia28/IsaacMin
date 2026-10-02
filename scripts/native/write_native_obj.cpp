#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <charconv>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <iostream>
struct Mapping{int fd;size_t size;void*data;Mapping(const char*path){fd=open(path,O_RDONLY);if(fd<0)throw std::runtime_error("inputopen");struct stat s;if(fstat(fd,&s))throw std::runtime_error("inputstat");size=s.st_size;data=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);if(data==MAP_FAILED)throw std::runtime_error("inputmap");}~Mapping(){munmap(data,size);close(fd);}};
int main(int argc,char**argv){try{if(argc!=8)return 2;Mapping a(argv[1]),b(argv[4]);size_t va=std::stoull(argv[2]),nv=std::stoull(argv[3]),fa=std::stoull(argv[5]),nf=std::stoull(argv[6]);if(va+nv*12>a.size||fa+nf*12>b.size)return 3;auto*v=(float*)((char*)a.data+va);auto*f=(int32_t*)((char*)b.data+fa);std::ofstream out(argv[7],std::ios::binary);if(!out)return 4;std::string buffer;buffer.reserve(4*1024*1024);char text[128];auto flush=[&](){out.write(buffer.data(),buffer.size());buffer.clear();};buffer+="# IsaacMin authoritative native float32 points and original triangle indices\no Terrain_FinalGround\n";for(size_t i=0;i<nv;i++){buffer+='v';for(int k=0;k<3;k++){double x=v[3*i+k];if(!std::isfinite(x))return 5;buffer+=' ';auto r=std::to_chars(text,text+128,x,std::chars_format::general,17);if(r.ec!=std::errc())return 6;buffer.append(text,r.ptr);}buffer+='\n';if(buffer.size()>2*1024*1024)flush();}for(size_t i=0;i<nf;i++){buffer+='f';for(int k=0;k<3;k++){int64_t x=f[3*i+k];if(x<0||size_t(x)>=nv)return 7;buffer+=' ';auto r=std::to_chars(text,text+128,x+1);if(r.ec!=std::errc())return 8;buffer.append(text,r.ptr);}buffer+='\n';if(buffer.size()>2*1024*1024)flush();}flush();if(!out)return 9;std::cout<<"vertices "<<nv<<" triangles "<<nf<<" exact_float32_decimal17\n";return 0;}catch(std::exception const&e){std::cerr<<e.what()<<"\n";return 10;}}
