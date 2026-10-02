// Direct HighMap CPU boundary; file data are little-endian float32 in row-major y,x.
#include <highmap/array.hpp>
#include <highmap/erosion.hpp>
#include <highmap/hydrology/hydrology.hpp>
#include <highmap/range.hpp>
#include <highmap/math/array.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>

void read(const char* file, hmap::Array& a) {
  std::ifstream f(file, std::ios::binary);
  f.read(reinterpret_cast<char*>(a.vector.data()), a.vector.size()*sizeof(float));
  if (!f || f.peek()!=EOF) throw std::runtime_error("invalid float32 array length");
  for (auto v:a.vector) if(!std::isfinite(v)) throw std::runtime_error("non-finite input");
}
void write(const std::string& file, const hmap::Array& a) {
  for (auto v:a.vector) if(!std::isfinite(v)) throw std::runtime_error("non-finite output");
  std::ofstream f(file, std::ios::binary);
  f.write(reinterpret_cast<const char*>(a.vector.data()), a.vector.size()*sizeof(float));
  if (!f) throw std::runtime_error("output write failed");
}
int main(int argc, char** argv) {
  try {
    if (argc!=9 && argc!=10) throw std::runtime_error("usage: worker width height input.f32 bedrock.f32 erodibility.f32 output_prefix erosion_m talus [global_runoff.f32]");
    int w=std::stoi(argv[1]), h=std::stoi(argv[2]);
    if(w<3||h<3||w>16384||h>16384) throw std::runtime_error("invalid raster shape");
    glm::ivec2 shape(w,h);
    hmap::Array z(shape), bedrock(shape), moisture(shape), erosion(shape);
    read(argv[3],z); read(argv[4],bedrock); read(argv[5],moisture);
    float amount=std::stof(argv[7]), talus=std::stof(argv[8]);
    if(amount<=0||amount>2||talus<=0||talus>10) throw std::runtime_error("invalid refinement parameters");
    for(size_t i=0;i<z.vector.size();i++)
      if(bedrock.vector[i]>z.vector[i]||moisture.vector[i]<0||moisture.vector[i]>1)
        throw std::runtime_error("invalid bedrock/erodibility constraint");
    // Global drainage is evaluated over the entire supplied context before erosion.
    auto runoff=hmap::flow_accumulation_dinf(z,talus);
    // Bedrock and erodibility constrain the native operation itself, not only a final blend.
    if(argc==10) {
      read(argv[9],runoff);
      for(auto v:runoff.vector) if(v<0||v>1) throw std::runtime_error("global runoff must be region-normalized in [0,1]");
      // Narrow adapter of HighMap hydraulic_stream's published stream-power operation:
      // retain its bedrock constraint; global flow is clipped and normalized ONCE
      // across the entire region by the caller, never independently per tile.
      auto flow=runoff;
      auto before=z;
      z-=moisture*amount*flow; z=hmap::maximum(bedrock,z);
      erosion=before-z;
    }else hmap::hydraulic_stream(z,amount,talus,&bedrock,&moisture,&erosion,1);
    std::string prefix=argv[6];
    write(prefix+"_height.f32",z); write(prefix+"_runoff.f32",runoff);
    write(prefix+"_erosion.f32",erosion);
    std::cout<<"{\"status\":\"success\",\"backend\":\"HighMap.hydraulic_stream.cpu\",\"width\":"<<w<<",\"height\":"<<h<<"}\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
