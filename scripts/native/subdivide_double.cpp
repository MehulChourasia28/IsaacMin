#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/topologyRefinerFactory.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/sdc/options.h>
#include <memory>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>
namespace Far=OpenSubdiv::Far;namespace Sdc=OpenSubdiv::Sdc;
struct Point{double x[3];void Clear(){x[0]=x[1]=x[2]=0;}void AddWithWeight(Point const&p,double w){for(int i=0;i<3;i++)x[i]+=p.x[i]*w;}};
void header(std::ofstream &f,const char*dtype,size_t n,int width){std::string h="{'descr': '"+std::string(dtype)+"', 'fortran_order': False, 'shape': ("+std::to_string(n)+", "+std::to_string(width)+"), }";size_t pad=(16-(10+h.size()+1)%16)%16;h+=std::string(pad,' ')+"\n";f.write("\223NUMPY\1\0",8);uint16_t len=h.size();f.write((char*)&len,2);f<<h;}
int main(int argc,char**argv){
 if(argc!=5)return 2;std::ifstream in(argv[1]);std::string magic;in>>magic;int nv=0,nf=0,ne=0;in>>nv>>nf>>ne;if(!in||magic!="OFF"||nv<3||nf<1)return 3;
 std::vector<Point> previous(nv);for(auto&p:previous){in>>p.x[0]>>p.x[1]>>p.x[2];for(double x:p.x)if(!std::isfinite(x))return 3;}if(!in)return 3;
 std::vector<int> counts(nf),indices;indices.reserve(size_t(3)*nf);for(int i=0;i<nf;i++){in>>counts[i];if(!in||counts[i]!=3)return 3;for(int j=0;j<counts[i];j++){int k=-1;in>>k;if(!in||k<0||k>=nv)return 3;indices.push_back(k);}}
 int levels=std::stoi(argv[2]);if(levels<1||levels>3)return 4;
 Far::TopologyDescriptor desc;desc.numVertices=nv;desc.numFaces=nf;desc.numVertsPerFace=counts.data();desc.vertIndicesPerFace=indices.data();
 Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Options options(Sdc::SCHEME_BILINEAR);
 std::unique_ptr<Far::TopologyRefiner> topology(Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(desc,options));if(!topology)return 5;
 Far::TopologyRefiner::UniformOptions uniform(levels);uniform.fullTopologyInLastLevel=true;topology->RefineUniform(uniform);
 Far::PrimvarRefinerReal<double> refine(*topology);
 for(int level=1;level<=levels;level++){std::vector<Point> next(topology->GetLevel(level).GetNumVertices());refine.Interpolate(level,previous,next);previous.swap(next);std::cout<<"level "<<level<<" vertices "<<previous.size()<<" faces "<<topology->GetLevel(level).GetNumFaces()<<std::endl;}
 auto& final=topology->GetLevel(levels);std::ofstream vertices(argv[3],std::ios::binary);header(vertices,"<f4",previous.size(),3);for(auto&p:previous){float x[3]={(float)p.x[0],(float)p.x[1],(float)p.x[2]};vertices.write((char*)x,sizeof(x));}vertices.close();
 std::ofstream faces(argv[4],std::ios::binary);header(faces,"<i4",final.GetNumFaces(),4);for(int i=0;i<final.GetNumFaces();i++){auto face=final.GetFaceVertices(i);if(face.size()!=4)return 6;for(int j=0;j<4;j++){int32_t x=face[j];faces.write((char*)&x,4);}}faces.close();
 if(!vertices||!faces)return 7;std::cout<<"complete\n";return 0;
}
