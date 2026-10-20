import * as THREE from 'three';
import {BUILDINGS,cameraAt,breathRay,clamp} from './game.js';
const WIDTH=1280,HEIGHT=704;
export function createScene(){
const renderer=new THREE.WebGLRenderer({canvas:document.querySelector('#film'),antialias:true,preserveDrawingBuffer:true});
renderer.setSize(WIDTH,HEIGHT);renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;
renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
const scene=new THREE.Scene();scene.background=new THREE.Color('#c5cbcc');scene.fog=new THREE.Fog('#c5cbcc',62,145);
const camera=new THREE.PerspectiveCamera(57,WIDTH/HEIGHT,.1,220);
scene.add(new THREE.HemisphereLight('#fff7e9','#77817b',2.2));
const sun=new THREE.DirectionalLight('#fff1d8',2);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);
Object.assign(sun.shadow.camera,{left:-45,right:45,top:45,bottom:-45,near:1,far:100});sun.shadow.normalBias=.04;scene.add(sun,sun.target);
const mats=new Map();function mat(c){if(!mats.has(c))mats.set(c,new THREE.MeshLambertMaterial({color:c,flatShading:true}));return mats.get(c);}
function mesh(parent,geo,c,pos){const m=new THREE.Mesh(geo,typeof c==='string'?mat(c):c);m.position.set(...pos);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
function box(parent,c,pos,size){return mesh(parent,new THREE.BoxGeometry(...size),c,pos);}
function lump(parent,c,pos,size){const m=mesh(parent,new THREE.IcosahedronGeometry(1,1),c,pos);m.scale.set(...size);return m;}
box(scene,'#8e9996',[0,-.25,-10],[320,.5,320]).castShadow=false;
for(let n=-77;n<=91;n+=14){box(scene,'#757f7e',[n,.006,-7],[6,.012,196]).castShadow=false;box(scene,'#757f7e',[7,.008,n],[196,.012,6]).castShadow=false;}
const city=BUILDINGS.map(b=>{
  const g=new THREE.Group();g.position.set(b.x,0,b.z);scene.add(g);
  const color=b.target?'#b68960':['#acb5b7','#b9b6ad','#94a4ad','#b9bfc0'][b.tone];
  const material=new THREE.MeshLambertMaterial({color,flatShading:true});
  const intact=new THREE.Group();g.add(intact);box(intact,material,[0,b.h/2,0],[b.w,b.h,b.d]);
  box(intact,'#839395',[0,b.h+.16,0],[b.w+.18,.32,b.d+.18]);
  for(let y=2;y<b.h-1;y+=2.7){
    for(const side of [-1,1]){
      box(intact,'#70868e',[0,y,side*(b.d/2+.015)],[b.w*.73,.6,.035]);
      box(intact,'#70868e',[side*(b.w/2+.015),y,0],[.035,.6,b.d*.73]);
    }
  }
  const debris=new THREE.Group();g.add(debris);const pieces=[];
  for(let iy=0;iy<4;iy++)for(let ix=0;ix<3;ix++)for(let iz=0;iz<2;iz++){
    const local=[(ix-1)*b.w/3,(iy+.5)*b.h/4,(iz-.5)*b.d/2];
    const m=box(debris,material,local,[b.w/3-.035,b.h/4-.04,b.d/2-.04]);
    const a=b.seed+ix*2.7+iz*3.8+iy*1.9;
    pieces.push({m,local,vx:Math.sin(a)*(1.4+iy*.7),vz:Math.cos(a)*(1.4+iy*.7),spin:Math.sin(a*2),floor:b.h/8+.18+(iy%2)*.18});
  }
  const dust=new THREE.Group();g.add(dust);const dustM=new THREE.MeshBasicMaterial({color:'#b4afa3',transparent:true,opacity:0,depthWrite:false});
  for(let i=0;i<7;i++){const puff=lump(dust,dustM,[0,0,0],[1,1,1]);puff.castShadow=false;}
  let turret=null,beacon=null;
  if(b.target){
    box(intact,'#614b3e',[0,b.h+.6,0],[2.5,1.1,2.5]);
    turret=mesh(intact,new THREE.CylinderGeometry(.4,.4,1,8),'#5e6765',[0,b.h+1,0]);
    beacon=lump(intact,new THREE.MeshBasicMaterial({color:'#ffb55c'}),[b.w/2-.8,b.h+1,0],[.55,.75,.55]);
  }
  return {b,g,intact,material,debris,pieces,dust,dustM,turret,beacon};
});

const monster=new THREE.Group();scene.add(monster);
const body=new THREE.Group();monster.add(body);
lump(body,'#54675d',[0,5.1,.6],[2.6,3.7,2.05]);
lump(body,'#819080',[0,4.9,-1.02],[1.7,2.5,.9]);
const neck=new THREE.Group();neck.position.set(0,8.8,-.65);body.add(neck);
lump(neck,'#586b61',[0,.35,0],[1.45,1.9,1.35]);
box(neck,'#5f7467',[0,.4,-1.16],[2.35,1.35,2.25]);
box(neck,'#293f38',[0,-.30,-1.75],[1.9,.18,1.25]);
const jaw=new THREE.Group();jaw.position.set(0,-.26,-.4);neck.add(jaw);box(jaw,'#53685b',[0,-.3,-1.25],[2.0,.52,1.8]);
for(const side of [-1,1]){
  box(neck,'#dec477',[side*1.2,.77,-1.65],[.08,.27,.47]);
  box(neck,'#26362e',[side*1.25,.78,-1.8],[.06,.19,.16]);
  for(let j=0;j<3;j++){const tooth=mesh(neck,new THREE.ConeGeometry(.14,.35,4),'#e0ddc3',[side*.81,-.32,-1.2-j*.43]);tooth.rotation.z=Math.PI;}
}
const legs=[],arms=[],tail=[];
for(const side of [-1,1]){
  const leg=new THREE.Group();leg.position.set(side*1.75,3.7,.6);monster.add(leg);
  lump(leg,'#52665a',[0,-1.2,0],[1.22,2.05,1.2]);box(leg,'#4c6255',[0,-3.1,-.55],[1.8,1.0,2.65]);
  for(let j=0;j<3;j++)box(leg,'#bcc1aa',[(j-1)*.52,-3.22,-1.95],[.35,.35,.55]);legs.push(leg);
  const arm=new THREE.Group();arm.position.set(side*2.0,6.85,-.35);body.add(arm);
  lump(arm,'#596e60',[side*.5,-.8,-.35],[.75,1.5,.75]);lump(arm,'#536858',[side*.55,-1.7,-1.25],[.58,.65,1.0]);arms.push(arm);
}
for(let i=0;i<9;i++){
  const t=i/8,m=lump(monster,'#53685b',[0,2.7-t*2,2.3+i*.85],[1.4*(1-t)+.18,1.15*(1-t)+.18,1.05]);tail.push(m);
}
const plates=[];
for(let i=0;i<8;i++){
  const y=2.4+i*.89,z=2.35-Math.max(0,i-3)*.23;
  const m=mesh(body,new THREE.ConeGeometry(.9,1.9,4),'#8b9c92',[0,y,z]);m.rotation.x=Math.PI/2;plates.push(m);
}
for(let i=0;i<5;i++){const m=mesh(monster,new THREE.ConeGeometry(.50-i*.065,1.1-i*.14,4),'#8b9c92',[0,3-i*.36,3.1+i*1.15]);m.rotation.x=.65;tail.push(m);}
const flame=new THREE.Group();scene.add(flame);const flameParts=[];
for(let i=0;i<24;i++){
  const m=mesh(flame,new THREE.IcosahedronGeometry(1,0),new THREE.MeshBasicMaterial({color:i%3===0?'#ffe4a0':i%3===1?'#ffad45':'#ed6c32'}),[0,0,0]);m.castShadow=false;flameParts.push(m);
}
const core=mesh(flame,new THREE.CylinderGeometry(1.3,.32,1,8,1,true),new THREE.MeshBasicMaterial({color:'#ffc269'}),[0,0,0]);core.castShadow=false;
const origin=new THREE.Vector3(),destination=new THREE.Vector3(),direction=new THREE.Vector3(),axis=new THREE.Vector3(0,1,0);
const shellMeshes=Array.from({length:32},()=>{const m=lump(scene,new THREE.MeshBasicMaterial({color:'#ffba55'}),[0,0,0],[.5,.5,.5]);m.castShadow=false;return m;});
const impactMeshes=Array.from({length:16},()=>{const m=lump(scene,new THREE.MeshBasicMaterial({color:'#ffdf93',transparent:true,opacity:.8}),[0,0,0],[1,1,1]);m.castShadow=false;return m;});
const shockwave=mesh(scene,new THREE.TorusGeometry(1,.075,6,72),new THREE.MeshBasicMaterial({color:'#edc887',transparent:true,opacity:0,depthWrite:false}),[0,.15,0]);shockwave.rotation.x=-Math.PI/2;shockwave.castShadow=false;
function render(s){
 const t=s.time,p=[s.x,0,s.z],strength=s.breathStrength;
 monster.position.set(...p);monster.rotation.y=-s.yaw;
 const moving=Math.hypot(...s.move)>.01,stompAge=t-s.stompAt,stomp=stompAge>=0&&stompAge<.6?Math.sin(stompAge/.6*Math.PI):0;
 body.position.y=s.dead?-2.5:(moving?Math.abs(Math.sin(s.stride))*.13:0)-stomp*.65;body.rotation.z=s.dead?.45:moving?Math.sin(s.stride)*.035:0;
 legs.forEach((l,i)=>{const stride=moving?Math.sin(s.stride+i*Math.PI):0;l.rotation.x=stride*.28;l.position.y=3.7+Math.abs(stride)*.48+(i===0?stomp*.5:0);});
 arms.forEach((a,i)=>{a.rotation.x=(moving?Math.sin(s.stride+i*Math.PI)*.12:0)-.22;});
 tail.forEach((m,i)=>{m.position.x=Math.sin(t*2.1-i*.34)*Math.min(i*.13,.7)*(s.dead?0:1);});
 neck.rotation.y=0;neck.rotation.x=-s.pitch;jaw.rotation.x=strength*.38;
 plates.forEach(m=>m.material.color.set(s.overheated?'#ff8156':strength>.1?'#ddc895':'#8b9c92'));
 monster.updateMatrixWorld(true);origin.set(0,-.28,-2.36);neck.localToWorld(origin);
 const ray=s.breath||breathRay(s);destination.set(ray.end.x,ray.end.y,ray.end.z);direction.copy(destination).sub(origin);const length=direction.length();direction.normalize();
 flame.visible=strength>0&&!s.dead&&length>.01;
 if(flame.visible){
  core.position.copy(origin).lerp(destination,.5);core.quaternion.setFromUnitVectors(axis,direction);core.scale.set(strength,length,strength);
  flameParts.forEach((m,i)=>{const u=(i/24+t*1.4)%1,spread=(.35+u*1.05)*strength;m.position.copy(origin).addScaledVector(direction,length*u);m.position.x+=Math.sin(i*2.8+t*18)*spread*.3;m.position.y+=Math.cos(i*4+t*12)*spread*.25;m.scale.set(spread*(1+Math.sin(i+t*21)*.15),spread*.85,spread);m.rotation.set(t*4+i,t*3,i);});
 }
 for(const v of city){
  const b=s.buildings[v.b.id],age=b.collapseAt===null?-1:t-b.collapseAt;
  v.intact.visible=b.hp>0;v.debris.visible=b.hp<=0;v.dust.visible=b.hp<=0&&age<3;
  const base=b.target?'#b68960':['#acb5b7','#b9b6ad','#94a4ad','#b9bfc0'][b.tone];v.material.color.set(b.hp<b.maxHp*.6?'#8b7970':base);v.material.emissive.set(t-b.burnAt<.6?'#70320d':'#000000');
  v.intact.rotation.z=b.hp>0&&t-b.hitAt<.15?Math.sin(t*43)*.012:0;
  if(v.turret){const from=new THREE.Vector3(0,b.h+1,0),aim=b.aim||{x:s.x,y:4.5,z:s.z},dir=new THREE.Vector3(aim.x-b.x,aim.y-from.y,aim.z-b.z).normalize();v.turret.position.copy(from).addScaledVector(dir,1.25);v.turret.quaternion.setFromUnitVectors(axis,dir);v.turret.scale.set(1,2.5,1);v.beacon.material.color.set(b.chargeAt!==null?'#ff573e':'#ffb55c');v.beacon.scale.setScalar(b.chargeAt!==null?1+Math.sin(t*16)*.12:1);}
  if(b.hp<=0){const elapsed=Math.min(age,2.4),spread=1-Math.exp(-elapsed*1.8);
   for(const q of v.pieces){q.m.position.set(q.local[0]+q.vx*spread,Math.max(q.floor,q.local[1]-5*elapsed*elapsed),q.local[2]+q.vz*spread);q.m.rotation.set(q.spin*Math.min(elapsed,1.3)*.55,Math.sin(q.spin)*Math.min(elapsed,1.3),q.spin*Math.min(elapsed,1.3)*.32);}
   v.dustM.opacity=Math.max(0,.32*(1-age/3));v.dust.children.forEach((m,i)=>{const a=i/7*Math.PI*2;m.position.set(Math.cos(a)*(2+age*1.7),.8+age*.9,Math.sin(a)*(2+age*1.7));m.scale.setScalar(1+Math.min(age,2)*1.6);});
  }
 }
 shellMeshes.forEach((m,i)=>{const p=s.shells[i];m.visible=!!p;if(p)m.position.set(p.x,p.y,p.z);});
 impactMeshes.forEach((m,i)=>{const p=s.impacts[i];m.visible=!!p;if(p){const age=t-p.time;m.position.set(p.x,p.y,p.z);m.scale.setScalar(.7+age*3);m.material.opacity=Math.max(0,1-age/.5);}});
 shockwave.visible=stompAge>=0&&stompAge<.6;if(shockwave.visible){shockwave.position.set(s.x,.2,s.z);shockwave.scale.setScalar(1+stompAge/.6*9);shockwave.material.opacity=.7*(1-stompAge/.6);}
 const view=cameraAt(s);camera.position.set(...view.position);camera.lookAt(...view.target);sun.position.set(s.x-22,48,s.z+18);sun.target.position.set(s.x,0,s.z);sun.target.updateMatrixWorld();renderer.render(scene,camera);
}
return{render,capture:()=>renderer.domElement.toDataURL('image/png'),captureBlob:()=>new Promise((resolve,reject)=>renderer.domElement.toBlob(b=>b?resolve(b):reject(Error('Unable to capture condition image')),'image/png')),camera,renderer};
}
export function drawHUD(canvas,s,label='LIVE',{gameUI=true,actions=true}={}){
 const c=canvas.getContext('2d');c.clearRect(0,0,canvas.width,canvas.height);if(!s)return;c.font='16px ui-monospace,monospace';c.textAlign='left';
 if(gameUI){c.fillStyle='rgba(8,18,30,.85)';c.fillRect(24,22,400,156);c.fillStyle='#c5dbe0';c.fillText(label,40,47);c.fillStyle='#fff';c.fillText(s.won?'CITY DEFENSES DESTROYED':s.dead?'DEFEATED':'DEFENSE TOWERS '+s.targets+' / 6',40,74);c.fillText('HEALTH '+s.health+' / 100',40,100);c.fillStyle='#3e4a53';c.fillRect(40,110,365,7);c.fillStyle=s.health>35?'#84dcca':'#ff8069';c.fillRect(40,110,365*s.health/100,7);c.fillStyle='#fff';c.fillText('HEAT '+Math.floor(s.heat)+' / 100'+(s.overheated?'  COOLING':'  F: BREATH'),40,140);c.fillStyle='#3e4a53';c.fillRect(40,152,365,7);c.fillStyle='#ffb55c';c.fillRect(40,152,365*s.heat/100,7);
  c.fillStyle='rgba(8,18,30,.85)';c.fillRect(430,625,420,44);c.fillStyle='#fff';c.textAlign='center';c.fillText('SPACE: STOMP '+(s.time-s.stompAt>=3.5?'READY':(3.5-s.time+s.stompAt).toFixed(1)+'s'),640,652);c.textAlign='left';
  if(s.time-s.hurtAt<.25){c.strokeStyle='#ff8059aa';c.lineWidth=10;c.strokeRect(5,5,1270,694);}
 }
 if(actions){const u=s.input,keys=[['W',u.forward>0],['A',u.side<0],['S',u.forward<0],['D',u.side>0],['F',u.fire],['SPACE',u.jump]];keys.forEach(([key,on],i)=>{const x=24+i*64;c.fillStyle=on?'#edb67d':'rgba(8,18,30,.85)';c.fillRect(x,621,58,48);c.fillStyle=on?'#15232b':'#e7f2f4';c.fillText(key,x+5,651);});}
}
