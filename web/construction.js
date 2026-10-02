import * as THREE from '/vendor/three.module.min.js';

// A diagram made from the build's real terrain and native USD samples.
// All final-world viewing remains in saved Isaac captures.
export class ConstructionPreview {
  constructor(host) {
    this.host=host;this.seen=new Set();this.reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.auto=!this.reduced;this.dirty=true;this.angle=-1.1;this.elevation=.78;this.distance=430;this.span=256;this.target=new THREE.Vector3(0,0,50);
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#132b29');
    this.camera=new THREE.PerspectiveCamera(42,1,.5,4000);this.camera.up.set(0,0,1);
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;host.append(this.renderer.domElement);
    this.scene.add(new THREE.HemisphereLight(0xe2fff0,0x4b5755,2));
    const sun=new THREE.DirectionalLight(0xfff4da,2.2);sun.position.set(-100,-140,240);this.scene.add(sun);
    this.group=new THREE.Group();this.scene.add(this.group);
    this.resize=new ResizeObserver(()=>this.size());this.resize.observe(host);this.size();
    let drag=null;const canvas=this.renderer.domElement;
    canvas.setAttribute('aria-label','Interactive construction preview. Drag to orbit and scroll to zoom.');
    canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);this.auto=false;};
    canvas.onpointermove=e=>{if(!drag)return;this.angle-=(e.clientX-drag[0])*.005;this.elevation=THREE.MathUtils.clamp(this.elevation+(e.clientY-drag[1])*.003,.18,1.4);drag=[e.clientX,e.clientY];this.dirty=true;};
    canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;
    canvas.onwheel=e=>{e.preventDefault();this.distance=THREE.MathUtils.clamp(this.distance*Math.exp(e.deltaY*.001),this.span*.4,this.span*4);this.dirty=true;};
    this.frame=this.frame.bind(this);this.frame();
  }
  size(){const box=this.host.getBoundingClientRect();if(!box.width||!box.height)return;this.camera.aspect=box.width/box.height;this.camera.updateProjectionMatrix();this.renderer.setSize(box.width,box.height);this.dirty=true;}
  clear(object){if(!object)return;this.group.remove(object);object.traverse(o=>{o.geometry?.dispose();if(o.material)for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();});}
  async update(previews){
    const surface=previews.terrain||previews.source;
    if(surface&&!this.seen.has(surface)){this.seen.add(surface);try{const r=await fetch(surface);if(!r.ok)throw Error('Terrain preview unavailable');this.setSurface(await r.json());}catch(e){this.seen.delete(surface);throw e;}}
    if(previews.assets&&!this.seen.has(previews.assets)){this.seen.add(previews.assets);try{const r=await fetch(previews.assets);if(!r.ok)throw Error('Asset preview unavailable');const data=await r.json();const points=await fetch(previews.assets.replace('assets.json','assets.bin'));if(!points.ok)throw Error('Asset point data unavailable');this.setAssets(data,await points.arrayBuffer());}catch(e){this.seen.delete(previews.assets);throw e;}}
  }
  setSurface(data){
    this.clear(this.terrain);this.clear(this.water);this.clear(this.grid);
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(data.positions.flat(),3));
    geometry.setAttribute('color',new THREE.Float32BufferAttribute(data.colors.flat(),3));
    const indices=[];for(let z=0;z<data.rows-1;z++)for(let x=0;x<data.columns-1;x++){const a=z*data.columns+x,b=a+1,c=a+data.columns,d=c+1;indices.push(a,c,b,b,c,d);}
    geometry.setIndex(indices);geometry.computeVertexNormals();geometry.computeBoundingBox();
    const material=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.94,metalness:.08,side:THREE.DoubleSide});
    this.terrain=new THREE.Mesh(geometry,material);this.group.add(this.terrain);
    const wire=new THREE.Mesh(geometry.clone(),new THREE.MeshBasicMaterial({color:0xb8e7bc,wireframe:true,transparent:true,opacity:.065}));this.terrain.add(wire);
    const box=geometry.boundingBox,size=box.getSize(new THREE.Vector3());this.target=box.getCenter(new THREE.Vector3());this.target.z=box.min.z+size.z*.35;
    this.base=box.min.z;this.surfaceBorn=performance.now();this.span=Math.max(size.x,size.y);this.distance=this.span*1.65;this.camera.far=Math.max(4000,this.span*10);this.camera.updateProjectionMatrix();this.dirty=true;
    this.grid=new THREE.GridHelper(Math.max(size.x,size.y)*1.5,24,0x537a64,0x294f42);this.grid.rotation.x=Math.PI/2;this.grid.position.set(this.target.x,this.target.y,this.base-4);this.group.add(this.grid);
    const wet=data.water_indices.map((index,n)=>[data.positions[index][0],data.positions[index][1],data.water[n]+.15]).flat();
    if(wet.length){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(wet,3));this.water=new THREE.Points(g,new THREE.PointsMaterial({color:0x72c8da,size:2.4,transparent:true,opacity:.8}));this.group.add(this.water);}
    this.host.dataset.surface=data.status;this.host.dataset.terrainVertices=String(data.positions.length);
  }
  setAssets(data,buffer){
    this.clear(this.assets);const geometry=new THREE.BufferGeometry();const packed=new THREE.InterleavedBuffer(new Float32Array(buffer),8);
    if(packed.count!==data.points)throw Error('Incomplete construction point data');
    geometry.setAttribute('position',new THREE.InterleavedBufferAttribute(packed,3,0));
    geometry.setAttribute('rootHeight',new THREE.InterleavedBufferAttribute(packed,1,3));
    geometry.setAttribute('revealOrder',new THREE.InterleavedBufferAttribute(packed,1,4));
    geometry.setAttribute('tint',new THREE.InterleavedBufferAttribute(packed,3,5));
    const material=new THREE.ShaderMaterial({transparent:true,depthWrite:false,uniforms:{age:{value:0},pixelRatio:{value:this.renderer.getPixelRatio()}},
      vertexShader:`attribute float rootHeight; attribute float revealOrder; attribute vec3 tint;
        uniform float age; uniform float pixelRatio; varying vec3 tone; varying float alpha;
        void main(){float a=smoothstep(0.,1.,age-revealOrder*7.); vec3 p=position; p.z=mix(rootHeight,position.z,a);
          vec4 mv=modelViewMatrix*vec4(p,1.); gl_Position=projectionMatrix*mv;
          gl_PointSize=clamp(650./max(1.,-mv.z),1.2,5.)*pixelRatio; tone=tint*(1.+.6*(1.-a));alpha=a;}`,
      fragmentShader:`varying vec3 tone; varying float alpha; void main(){float d=length(gl_PointCoord-.5);if(d>.5||alpha<.01)discard;gl_FragColor=vec4(tone,alpha*(1.-smoothstep(.32,.5,d)));}`});
    this.assets=new THREE.Points(geometry,material);this.assets.frustumCulled=false;this.group.add(this.assets);this.assetsBorn=performance.now();this.dirty=true;
    this.host.dataset.assetPoints=String(data.points);this.host.dispatchEvent(new CustomEvent('construction-assets',{detail:data}));
  }
  frame(now=performance.now()){
    if(this.disposed)return;this.request=requestAnimationFrame(this.frame);if(document.hidden||this.host.closest('[hidden]'))return;
    if(this.last&&now-this.last<40)return;const delta=this.last?Math.min(.1,(now-this.last)/1000):0;this.last=now;
    const growing=!this.reduced&&((this.terrain&&now-this.surfaceBorn<2200)||(this.assets&&now-this.assetsBorn<8200));
    if(!this.auto&&!growing&&!this.dirty)return;this.dirty=false;
    if(this.auto)this.angle+=delta*.045;
    if(this.terrain){const t=this.reduced?1:Math.min(1,(now-this.surfaceBorn)/2000),grow=1-Math.pow(1-t,3);this.terrain.scale.z=grow;this.terrain.position.z=this.base*(1-grow);}
    if(this.assets)this.assets.material.uniforms.age.value=this.reduced?20:(now-this.assetsBorn)/1000;
    this.camera.position.set(this.target.x+Math.cos(this.angle)*this.distance*Math.cos(this.elevation),this.target.y+Math.sin(this.angle)*this.distance*Math.cos(this.elevation),this.target.z+this.distance*Math.sin(this.elevation));this.camera.lookAt(this.target);this.renderer.render(this.scene,this.camera);
  }
  dispose(){this.disposed=true;cancelAnimationFrame(this.request);this.resize.disconnect();for(const item of [this.terrain,this.water,this.grid,this.assets])this.clear(item);this.renderer.dispose();this.renderer.domElement.remove();}
}
