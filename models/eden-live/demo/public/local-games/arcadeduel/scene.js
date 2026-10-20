import * as THREE from 'three';
import {MOVES,clamp} from './game.js';
const WIDTH=1280,HEIGHT=704;
export function createScene(){
const renderer=new THREE.WebGLRenderer({canvas:document.querySelector('#film'),antialias:true,preserveDrawingBuffer:true});renderer.setSize(WIDTH,HEIGHT);renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
const scene=new THREE.Scene();scene.background=new THREE.Color('#11172f');scene.fog=new THREE.Fog('#11172f',15,35);const camera=new THREE.OrthographicCamera(-5.3,5.3,2.915,-2.915,.1,60);
scene.add(new THREE.HemisphereLight('#d8efff','#496176',1.85));const light=new THREE.DirectionalLight('#fff0d5',3.0);light.position.set(-3,8,6);light.castShadow=true;light.shadow.mapSize.set(2048,2048);Object.assign(light.shadow.camera,{left:-7,right:7,top:6,bottom:-6,near:1,far:25});light.shadow.normalBias=.015;scene.add(light);
const rim=new THREE.DirectionalLight('#69d3ed',1.25);rim.position.set(3,4,-4);scene.add(rim);
const mats=new Map(),Y=new THREE.Vector3(0,1,0),V=a=>new THREE.Vector3(...a);
function mat(c){if(!mats.has(c))mats.set(c,new THREE.MeshLambertMaterial({color:c,flatShading:true}));return mats.get(c);}
function mesh(g,geo,c,p=[0,0,0]){const o=new THREE.Mesh(geo,typeof c==='string'?mat(c):c);o.position.set(...p);o.castShadow=true;o.receiveShadow=true;g.add(o);return o;}
function box(g,c,p,size){return mesh(g,new THREE.BoxGeometry(...size),c,p);}
function group(g,p=[0,0,0]){const o=new THREE.Group();o.position.set(...p);g.add(o);return o;}
function ball(g,c,p,scale){const o=mesh(g,new THREE.IcosahedronGeometry(1,1),c,p);o.scale.set(...scale);return o;}
function rod(g,c,a,b,r){const o=mesh(g,new THREE.CylinderGeometry(r,r,1,8),c);segment(o,V(a),V(b));return o;}
function segment(o,a,b){o.position.copy(a).add(b).multiplyScalar(.5);o.scale.y=a.distanceTo(b);o.quaternion.setFromUnitVectors(Y,b.clone().sub(a).normalize());}
box(scene,'#315365',[0,-.18,0],[9.2,.35,4]);box(scene,'#467d91',[0,.004,0],[8.9,.025,3.75]);box(scene,'#203c50',[0,-.46,0],[40,.2,30]);
for(const x of [-3.4,3.4])box(scene,x<0?'#208dce':'#ed8437',[x,.020,0],[.11,.012,3.5]);
for(const x of [-5.4,0,5.4]){box(scene,'#254254',[x,2,-5],[2.6,4,.35]);box(scene,'#3c7081',[x,3.3,-4.79],[2.2,.10,.03]);}
const fighters=[0,1].map(id=>{
 const root=group(scene),body=group(root),color=id?'#fa701c':'#008ee8',dark=id?'#a84823':'#245380',skin=id?'#cf9470':'#e2ac81';root.rotation.y=id?Math.PI:0;
 box(body,color,[0,1.32,0],[.43,.64,.57]);box(body,dark,[-.04,.91,0],[.40,.25,.54]);box(body,'#e8dcc1',[.005,1.0,0],[.43,.075,.56]);
 const head=group(body,[.025,1.86,0]);ball(head,skin,[0,0,0],[.24,.27,.245]);box(head,dark,[0,.19,0],[.37,.12,.41]);box(head,color,[.185,.11,0],[.07,.06,.46]);ball(head,skin,[.238,-.005,0],[.052,.070,.07]);
 for(const z of [-.12,.12])box(head,'#233241',[.227,.053,z],[.016,.037,.033]);
 const arms=[-1,1].map(zside=>({zside,hand:ball(body,id?'#ff932e':'#159cde',[0,0,0],[.17,.17,.18]),upper:rod(body,color,[0,0,0],[0,1,0],.102),fore:rod(body,skin,[0,0,0],[0,1,0],.084)}));
 const legs=[-1,1].map(zside=>{const foot=group(body);box(foot,dark,[.065,.025,0],[.36,.20,.23]);box(foot,'#d7e3d9',[.07,-.084,0],[.38,.025,.25]);return{zside,foot,upper:rod(body,dark,[0,0,0],[0,1,0],.125),lower:rod(body,color,[0,0,0],[0,1,0],.10)};});
 return{id,root,body,head,arms,legs};
});
function ik(a,b,len,reference){const d=b.clone().sub(a),distance=d.length(),axis=d.clone().normalize();let bend=reference.clone().addScaledVector(axis,-reference.dot(axis));if(bend.lengthSq()<.001)bend.set(1,0,0);bend.normalize();return a.clone().add(b).multiplyScalar(.5).addScaledVector(bend,Math.sqrt(Math.max(.008,len*len-distance*distance/4)));}
function armPose(f,a,point){a.hand.position.copy(point);a.hand.rotation.set(0,0,-.12);const shoulder=V([.08,1.52,a.zside*.25]),elbow=ik(shoulder,point,.52,V([0,-1,0]));segment(a.upper,shoulder,elbow);segment(a.fore,elbow,point);return shoulder.distanceTo(point);}
function legPose(f,l,point,kicking=false){l.foot.position.copy(point);l.foot.rotation.z=kicking?-.05:0;const hip=V([0,.91,l.zside*.18]),knee=ik(hip,point,kicking?.5:.415,V([1,0,0]));if(Math.abs(f.body.rotation.z)<.3)knee.y=Math.max(knee.y,.24-f.body.position.y);l.kneeAngle=knee.clone().sub(hip).angleTo(point.clone().sub(knee))*180/Math.PI;segment(l.upper,hip,knee);segment(l.lower,knee,point);return hip.distanceTo(point);}
const sparkMat=new THREE.MeshBasicMaterial({color:'#fff4c9',transparent:true,opacity:1});
const sparks=Array.from({length:14},()=>mesh(scene,new THREE.ConeGeometry(.035,.21,4),sparkMat));
const waves=Array.from({length:8},()=>mesh(scene,new THREE.TorusGeometry(.48,.065,6,28,Math.PI*1.45),new THREE.MeshBasicMaterial({color:'#32d5ff'})));
const arc=mesh(scene,new THREE.TorusGeometry(.85,.035,6,36,Math.PI*1.3),new THREE.MeshBasicMaterial({color:'#45d4ff',transparent:true,opacity:.65}));
const warning=mesh(scene,new THREE.RingGeometry(.45,.50,32),new THREE.MeshBasicMaterial({color:'#ffb15b',side:THREE.DoubleSide,transparent:true,opacity:.8}));warning.rotation.x=-Math.PI/2;
const ghosts=fighters.map(f=>Array.from({length:3},(_,i)=>{const g=f.root.clone(true),m=new THREE.MeshBasicMaterial({color:f.id?'#ff8224':'#14bdff',transparent:true,opacity:.16-i*.04,depthWrite:false});g.traverse(o=>{if(o.isMesh){o.material=m;o.castShadow=false;}});scene.add(g);return g;}));
function render(s){
 const t=s.time;
 fighters.forEach((f,id)=>{
  const p=s.fighters[id],at=p.attack,m=at?MOVES[at.type]:null,wind=m?m.wind+(id?30:0):0;
  const q=at?(at.age<wind?Math.max(0,(at.age-wind+3)/3):Math.max(0,1-(at.age-wind)/(m.active+m.recovery))):0;
  const fall=p.koAt===null?0:clamp((t-p.koAt)/.42),recoil=Math.max(0,1-(t-p.hitAt)/.25),kick=at?.type==='spin',air=clamp(p.y*2);
  f.root.position.set(p.x,p.y,0);f.root.rotation.y=p.facing<0?Math.PI:0;
  f.body.position.set(0,fall*.32,0);f.body.rotation.set(0,kick?q*Math.PI*2:0,fall*Math.PI/2+recoil*.1);f.head.rotation.z=recoil*.1;
  f.arms.forEach((a,i)=>{let point=V([.3,1.5,a.zside*.27]);
   if(p.blocking)point.set(.42,1.72,a.zside*.17);
   if(at&&(!kick||at.age<wind)){
    if(at.type==='wave')point.set(.45+q*.3,1.33,a.zside*.13);
    else if(i===(p.combo%2)){point.lerp(V(at.type==='uppercut'?[.55,2.25,a.zside*.18]:[1.02,1.62,a.zside*.20]),q);}
    if(id&&at.age<wind)point.x-=.15;
   }
   if(fall)point.lerp(V([.2,.85,a.zside*.35]),fall);
   if((s.won||s.ending?.playerWon)&&id===0&&i===0)point.set(.16,2.18,.25);
   armPose(f,a,point);
  });
  f.legs.forEach((l,i)=>{const stride=Math.sin(t*14+i*Math.PI)*Math.abs(p.walk),point=V([(i?.2:-.2)+stride*.12,.12+Math.max(0,stride)*.1+air*(i?.2:.3),l.zside*.17]);if(kick&&i===1)point.lerp(V([.9,1.22,l.zside*.15]),q);legPose(f,l,point,kick);});
  f.root.updateMatrixWorld(true);
  ghosts[id].forEach((g,j)=>{g.visible=p.dash>0;if(!g.visible)return;const source=[];f.root.traverse(o=>source.push(o));let n=0;g.traverse(o=>{const src=source[n++];o.position.copy(src.position);o.quaternion.copy(src.quaternion);o.scale.copy(src.scale);});g.position.x-=(j+1)*.25*p.dashDir;g.position.z=-.05*(j+1);});
 });
 const hit=s.hits.at(-1),age=hit?t-hit.time:1;
 sparks.forEach((o,i)=>{o.visible=!!hit&&age<.25;if(!o.visible)return;const angle=i*2.399,r=.08+age*2.5;o.position.set(hit.x+Math.cos(angle)*r,hit.y+Math.sin(angle)*r,.4);o.rotation.z=angle;o.scale.setScalar((1-age/.25)*(hit.result==='block'?.5:1));});
 waves.forEach((o,i)=>{const w=s.waves[i];o.visible=!!w;if(w){o.position.set(w.x,w.y,.25);o.rotation.set(0,w.dir<0?Math.PI:0,-Math.PI*.7);o.material.color.set(w.owner?'#ffad45':'#37ceff');}});
 const active=s.fighters.find(p=>p.attack&&['spin','uppercut'].includes(p.attack.type)&&p.attack.age>=MOVES[p.attack.type].wind+(p.id?30:0)-2);
 arc.visible=!!active;if(active){arc.position.set(active.x,active.y+1.1,.35);arc.rotation.z=t*12;arc.material.color.set(active.id?'#ffac55':'#38d8ff');}
 const enemy=s.fighters[1];warning.visible=!!enemy.attack&&enemy.attack.age<MOVES[enemy.attack.type].wind+30;
 if(warning.visible){warning.position.set(enemy.x,.04,0);warning.scale.setScalar(1.2+Math.sin(t*12)*.12);}
 const height=Math.max(...s.fighters.map(f=>f.y));camera.position.set(0,3.2+height*.2,12);camera.lookAt(0,1.25+height*.25,0);camera.updateMatrixWorld();renderer.render(scene,camera);
}
return {render,capture:()=>renderer.domElement.toDataURL('image/png'),captureBlob:()=>new Promise((resolve,reject)=>renderer.domElement.toBlob(b=>b?resolve(b):reject(Error('Capture failed')),'image/png')),renderer,camera};
}
export function drawHUD(canvas,s,label='LIVE',{gameUI=true,actions=true}={}){
 const c=canvas.getContext('2d');c.clearRect(0,0,canvas.width,canvas.height);if(!s)return;c.font='18px monospace';c.textAlign='left';
 if(gameUI){for(const [id,x] of [[0,30],[1,850]]){const f=s.fighters[id];c.fillStyle='#101b2be6';c.fillRect(x,25,400,105);c.fillStyle=id?'#ffac63':'#69cbff';c.fillText((id?'ORANGE / CPU':'BLUE / YOU')+'  '+f.hp+' HP',x+16,53);c.fillStyle='#3e4d5a';c.fillRect(x+16,66,368,12);c.fillStyle=id?'#f59142':'#47b9f3';c.fillRect(x+16,66,368*f.hp/100,12);c.fillStyle='#d6e4ef';c.fillText('ENERGY '+Math.floor(f.energy)+'   GUARD '+Math.floor(f.guard),x+16,107);}
 c.textAlign='center';c.fillStyle='#fff';c.fillText(Math.max(0,60-Math.floor(s.time))+'s',640,55);if(s.combo>1)c.fillText(s.combo+' HIT COMBO',640,96);
 const e=s.fighters[1];if(e.attack&&e.attack.age<MOVES[e.attack.type].wind+30){c.fillStyle='#ffb968';c.fillText('INCOMING '+e.attack.type.toUpperCase()+' — GUARD OR MOVE',640,570);}
 }
 if(actions){c.textAlign='center';const u=s.input;[['A',u.side<0],['D',u.side>0],['F',u.fire],['G',u.guard],['E',u.upper],['R',u.wave],['SHIFT',u.dash],['SPACE',u.jump]].forEach(([key,on],i)=>{const x=280+i*90;c.fillStyle=on?'#77c6e8':'#102030dd';c.fillRect(x,625,82,42);c.fillStyle=on?'#142535':'#e4f0f5';c.fillText(key,x+41,652);});}
}
