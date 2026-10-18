import * as THREE from 'three';
export {THREE};
export const V=a=>new THREE.Vector3(...a),UP=new THREE.Vector3(0,1,0);
const mats=new Map();export function mat(c){if(!mats.has(c))mats.set(c,new THREE.MeshLambertMaterial({color:c,flatShading:true}));return mats.get(c);}
export function mesh(g,geo,c,p=[0,0,0]){const o=new THREE.Mesh(geo,typeof c==='string'?mat(c):c);o.position.set(...p);o.castShadow=true;o.receiveShadow=true;g.add(o);return o;}
export function group(g,p=[0,0,0]){const o=new THREE.Group();o.position.set(...p);g.add(o);return o;}
export function box(g,c,p,size){return mesh(g,new THREE.BoxGeometry(...size),c,p);}
export function ball(g,c,p,size,detail=0){const o=mesh(g,new THREE.IcosahedronGeometry(1,detail),c,p);o.scale.set(...size);return o;}
export function rod(g,c,a,b,r1,r2=r1){const A=V(a),B=V(b),o=mesh(g,new THREE.CylinderGeometry(r2,r1,A.distanceTo(B),8),c,A.clone().add(B).multiplyScalar(.5).toArray());o.quaternion.setFromUnitVectors(UP,B.sub(A).normalize());return o;}
export function segment(o,a,b){o.position.copy(a).add(b).multiplyScalar(.5);o.scale.y=a.distanceTo(b);o.quaternion.setFromUnitVectors(UP,b.clone().sub(a).normalize());}
export function createLavaWorld(width=1280,height=720){const renderer=new THREE.WebGLRenderer({canvas:document.querySelector('#film'),antialias:true,preserveDrawingBuffer:true});renderer.setSize(width,height);renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
 const scene=new THREE.Scene();scene.background=new THREE.Color('#262438');scene.fog=new THREE.Fog('#3e2c38',65,170);scene.add(new THREE.HemisphereLight('#c5d9ee','#e26b35',1.8));const sun=new THREE.DirectionalLight('#fff0d1',2.25);sun.position.set(-20,35,15);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-45,right:45,top:52,bottom:-52,near:1,far:145});sun.shadow.normalBias=.03;scene.add(sun,sun.target);const fill=new THREE.DirectionalLight('#a7dfff',.75);fill.position.set(15,20,-10);scene.add(fill);
 const positions=[],colors=[];for(let x=-95;x<95;x+=5)for(let z=-115;z<70;z+=5){const points=[[x,z],[x+5,z],[x,z+5],[x+5,z+5]],k=Math.abs(Math.floor(x/5)*17+Math.floor(z/5)*23)%9,c=new THREE.Color(k<2?'#ff9727':k<5?'#f37520':'#e55f1c');for(const id of [0,2,1,1,2,3]){positions.push(points[id][0],0,points[id][1]);colors.push(c.r,c.g,c.b);}}
 const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geo.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));const lava=mesh(scene,geo,new THREE.MeshBasicMaterial({vertexColors:true}));lava.castShadow=false;lava.receiveShadow=false;
 const crust=[];for(let i=0;i<170;i++){const x=-75+i*17.713%150,z=55-i*19.831%160,r=.3+(i%5)*.18;const p=mesh(scene,new THREE.CircleGeometry(r,5),'#65352f',[x,.012,z]);p.rotation.x=-Math.PI/2;p.rotation.z=i;crust.push({p,x,z});}
 for(let i=0;i<36;i++){const side=i%2?1:-1,x=side*(27+(i%4)*5),z=35-Math.floor(i/2)*8;const rock=ball(scene,i%3?'#3e4355':'#505467',[x,4+(i%3),z],[7,10+(i%4)*3,7],1);rock.rotation.y=i*.7;}
 const ceiling=box(scene,'#272b3c',[0,28,-24],[100,5,165]);ceiling.receiveShadow=false;for(let i=0;i<24;i++)mesh(scene,new THREE.ConeGeometry(1.3+(i%3)*.4,5+(i%4),5),'#363c50',[-26+(i*13.1%53),24,-75+(i*19.1%116)]).rotation.z=Math.PI;
 function animate(t,center=[0,0,-20]){for(const {p,x,z} of crust){p.position.x=x+Math.sin(t*.3+z)*.4;p.position.z=z+t*.08;}sun.position.set(center[0]-23,center[1]+38,center[2]+18);sun.target.position.fromArray(center);sun.target.updateMatrixWorld();}
 return{renderer,scene,sun,animate};}
