import * as THREE from 'three';
import {ENEMIES,POOL,smooth,clamp} from './game.js';
import {drawActions} from '../runtime/action-overlay.js';
const WIDTH=1280,HEIGHT=704;
export function createScene(){
const renderer=new THREE.WebGLRenderer({canvas:document.querySelector('#film'),antialias:true,preserveDrawingBuffer:true});renderer.setSize(WIDTH,HEIGHT);renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
const scene=new THREE.Scene();scene.background=new THREE.Color('#112c34');scene.fog=new THREE.Fog('#17323b',24,48);const camera=new THREE.PerspectiveCamera(74,WIDTH/HEIGHT,.045,80);scene.add(camera);scene.add(new THREE.HemisphereLight('#a8dfe0','#273e46',1.2));
const sun=new THREE.DirectionalLight('#ffd194',2.1);sun.position.set(-4,8,11);sun.target.position.set(0,0,-4);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-12,right:12,top:22,bottom:-18,near:.1,far:50});sun.shadow.normalBias=.025;scene.add(sun,sun.target);
const V=a=>new THREE.Vector3(...a),up=V([0,1,0]),mats=new Map(),colliders=[];
function mat(c){if(!mats.has(c))mats.set(c,new THREE.MeshLambertMaterial({color:c,flatShading:true}));return mats.get(c)}
const glow=c=>new THREE.MeshBasicMaterial({color:c});
function mesh(parent,geo,col,pos=[0,0,0]){const m=new THREE.Mesh(geo,typeof col==='string'?mat(col):col);m.position.fromArray(pos);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
function box(p,c,pos,size,solid=false){const m=mesh(p,new THREE.BoxGeometry(...size),c,pos);if(solid)colliders.push(m);return m;}
function group(p,pos=[0,0,0]){const g=new THREE.Group();g.position.fromArray(pos);p.add(g);return g;}
function ball(p,c,pos,r,detail=0){return mesh(p,new THREE.IcosahedronGeometry(r,detail),c,pos)}
function rod(p,a,b,r,c){const A=V(a),B=V(b),m=mesh(p,new THREE.CylinderGeometry(r,r,A.distanceTo(B),8),c,A.clone().add(B).multiplyScalar(.5).toArray());m.quaternion.setFromUnitVectors(up,B.sub(A).normalize());return m;}
function segment(m,A,B){const a=Array.isArray(A)?V(A):A,b=Array.isArray(B)?V(B):B,d=b.clone().sub(a);m.position.copy(a).add(b).multiplyScalar(.5);m.scale.set(1,Math.max(.0001,d.length()),1);m.quaternion.setFromUnitVectors(up,d.normalize());}
box(scene,'#182d35',[0,-.19,-.5],[16,.38,32]);
for(let ix=0;ix<8;ix++)for(let iz=0;iz<16;iz++)box(scene,(ix+iz)%2?'#3c686b':'#315258',[-7+ix*2,.013,14-iz*2],[1.94,.026,1.94]);
for(const x of [-7.7,7.7]){box(scene,'#184b51',[x,2.9,-.5],[.5,5.8,32],true);box(scene,'#a47244',[x*.968,1.0,-.5],[.12,.17,32]);}
box(scene,'#17434b',[0,2.9,-16],[16,5.8,.5],true);box(scene,'#17434b',[0,2.9,15.5],[16,5.8,.5],true);box(scene,'#15353b',[0,5.9,-.5],[16,.25,32]);
for(const z of [12,5,-2,-9,-15]){for(const x of [-7.2,7.2])box(scene,'#776447',[x,2.8,z],[.42,5.6,.48],true);box(scene,'#66573e',[0,5.52,z],[14.6,.38,.5]);box(scene,'#b98f58',[0,5.28,z],[14.2,.08,.58]);}
for(const x of [-5.6,5.6])rod(scene,[x,4.7,14],[x,4.7,-15],.18,'#ae7442');
for(const [x,z]of [[-6.4,8],[6.4,2],[-6.4,-10]]){rod(scene,[x,.2,z],[x,4.7,z],.2,'#b7824a');rod(scene,[x,2.0,z],[x>0?4.2:-4.8,2.0,z],.14,'#9c673e');const valve=mesh(scene,new THREE.TorusGeometry(.34,.07,6,12),'#c54e35',[x>0?5.0:-5.0,2,z+.05]);box(valve,'#ba4935',[0,0,0],[.60,.06,.05]);box(valve,'#ba4935',[0,0,0],[.06,.60,.05]);}
function lamp(x,z){box(scene,'#775b3c',[x,4.1,z],[.25,.85,.45]);box(scene,glow('#ffce70'),[x,4.1,z+.26],[.22,.62,.10]);const l=new THREE.PointLight('#ffc979',30,9,2);l.position.set(x,3.75,z+.7);scene.add(l);}
lamp(-5.7,6);lamp(5.6,-4);lamp(-5.5,-13);
// A lit submarine window and simple silhouettes provide the underwater cue.
box(scene,glow('#226c79'),[-7.42,2.85,-4.4],[.035,3.2,5.6]);
for(const z of [-7.3,-1.5])box(scene,'#c29a5b',[-7.35,2.85,z],[.2,3.65,.18]);for(const y of [1.05,4.65])box(scene,'#c29a5b',[-7.35,y,-4.4],[.2,.18,6]);
for(const z of [-5.5,-3.4])box(scene,'#8e744d',[-7.28,2.85,z],[.18,3.6,.11]);
for(let i=0;i<7;i++){const f=ball(scene,glow('#174d5a'),[-7.36,1.5+(i%3)*.84,-6.6+i*.69],.24);f.scale.set(.07,.34,1);const tail=mesh(scene,new THREE.ConeGeometry(.12,.24,3),glow('#174d5a'),[-7.36,f.position.y,f.position.z-.29]);tail.rotation.x=Math.PI/2;tail.scale.x=.08;}
// Large generator on the right of the walking route, two boilers and a rear door.
const gen=group(scene,[1.3,0,-2.1]);box(gen,'#26454d',[0,1.2,0],[3.4,2.4,3.0],true);box(gen,'#896d45',[0,2.46,0],[3.55,.12,3.12],true);for(const x of [-.8,.8]){mesh(gen,new THREE.CylinderGeometry(.5,.5,2.35,10),'#477173',[x,1.5,0]);for(let i=0;i<5;i++)box(gen,'#152f38',[x,.6+i*.38,1.525],[.61,.10,.07]);}
box(gen,'#a38550',[0,1.65,1.62],[.48,.72,.10]);box(gen,glow('#80cece'),[0,1.8,1.68],[.28,.23,.035]);
for(const[x,z]of[[-5.8,5.8],[5.8,-11.8]]){mesh(scene,new THREE.CylinderGeometry(.95,1,2.7,10),'#7d704c',[x,1.35,z]);for(const y of [.25,1.1,2.3])mesh(scene,new THREE.CylinderGeometry(1.03,1.03,.12,10),'#bb9359',[x,y,z]);box(scene,'#27474a',[x,1.1,z+.97],[.75,1.1,.09]);}
box(scene,'#745d3e',[.5,2,-15.69],[3.5,4,.12]);box(scene,'#284851',[.5,1.9,-15.56],[2.6,3.6,.10]);for(const x of [-.45,.5,1.45])box(scene,'#b79055',[x,1.9,-15.49],[.08,3.6,.05]);box(scene,glow('#7fe8b5'),[.5,4.15,-15.48],[1.7,.24,.06]);
const water=mesh(scene,new THREE.CircleGeometry(1,40),new THREE.MeshPhongMaterial({color:'#185970',transparent:true,opacity:.70,shininess:85,side:THREE.DoubleSide,depthWrite:false}),[POOL.x,.058,POOL.z]);water.rotation.x=-Math.PI/2;water.scale.set(POOL.rx,POOL.rz,1);
const poolLight=new THREE.PointLight('#37caff',0,10,2);poolLight.position.set(POOL.x,.4,POOL.z);scene.add(poolLight);
const rings=Array.from({length:3},()=>{const m=mesh(scene,new THREE.TorusGeometry(1,.013,5,48),new THREE.MeshBasicMaterial({color:'#55dfff',transparent:true,opacity:0,depthWrite:false}),[POOL.x,.088,POOL.z]);m.rotation.x=-Math.PI/2;return m;});
const flow=Array.from({length:36},()=>mesh(scene,new THREE.CylinderGeometry(.018,.018,1,5),glow('#54dfff')));
const enemies=ENEMIES.map(e=>{const root=group(scene,[e.x,.04,e.z]),body=group(root),legs=[];
 for(const side of [-1,1]){const l=group(body,[side*.17,.66,0]);box(l,'#263f4c',[0,-.29,0],[.25,.57,.28]);box(l,'#172f38',[0,-.60,-.09],[.3,.15,.40]);legs.push(l);}
 box(body,e.id%2?'#654f65':'#a0673f',[0,1.04,0],[.66,.74,.40]);box(body,'#303b3b',[0,1.08,-.24],[.48,.44,.12]);for(const x of [-.24,.24])box(body,'#b18b54',[x,1.10,-.26],[.06,.60,.05]);
 box(body,'#d3c5a0',[0,1.62,0],[.42,.46,.38]);box(body,'#223941',[0,1.66,-.208],[.32,.095,.05]);for(const x of [-.095,.095])ball(body,glow('#fbd58b'),[x,1.66,-.24],.029);box(body,'#74533c',[0,1.47,-.21],[.21,.11,.05]);
 const arms=[-1,1].map(side=>({side,upper:mesh(body,new THREE.CylinderGeometry(.12,.13,1,7),e.id%2?'#766474':'#b17a49'),lower:mesh(body,new THREE.CylinderGeometry(.10,.11,1,7),'#506165'),hand:ball(body,'#b8a27d',[0,0,0],.12)}));
 const arcs=Array.from({length:14},()=>mesh(scene,new THREE.CylinderGeometry(.014,.014,1,5),glow('#8cf5ff')));return {e,root,body,legs,arms,arcs};
});
const wrench=group(camera);box(wrench,'#b8492b',[0,.41,0],[.12,.83,.11]);box(wrench,'#f07838',[0,.54,-.02],[.085,.59,.14]);for(let i=0;i<5;i++)box(wrench,'#253b42',[0,.1+i*.054,0],[.15,.025,.14]);box(wrench,'#d77b3b',[0,.91,0],[.42,.24,.20]);for(const side of [-1,1]){box(wrench,'#f6bb68',[side*.173,1.045,0],[.095,.25,.21]);box(wrench,'#785239',[side*.14,1.11,-.015],[.03,.035,.23]);}box(wrench,'#75472d',[0,.83,-.115],[.18,.06,.035]);
const leftHand=group(camera);box(leftHand,'#c6a77e',[0,0,0],[.19,.085,.22]);for(let i=0;i<4;i++){const x=-.073+i*.049;rod(leftHand,[x,0,-.085],[x*.95,.02,-.22],.023,'#d7b58a');rod(leftHand,[x*.95,.02,-.22],[x*.88,.046,-.30+(i%2)*.028],.021,'#d7b58a');rod(leftHand,[x,.046,-.04],[x*.94,.055,-.22],.009,glow('#63daff'));}rod(leftHand,[-.09,-.005,.025],[-.18,.025,-.10],.034,'#c9aa80');
const arms=[-1,1].map(side=>({side,upper:mesh(camera,new THREE.CylinderGeometry(.095,.13,1,8),'#826548'),lower:mesh(camera,new THREE.CylinderGeometry(.082,.098,1,8),'#b69c70'),hand:side===1?ball(camera,'#c6a77e',[0,0,0],.10,1):null}));
const handLight=new THREE.PointLight('#36bfff',0,3,2);leftHand.add(handLight);handLight.position.set(0,.06,-.20);
const boltSegments=Array.from({length:30},()=>({core:mesh(scene,new THREE.CylinderGeometry(.021,.021,1,5),glow('#d4ffff')),glow:mesh(scene,new THREE.CylinderGeometry(.049,.049,1,5),new THREE.MeshBasicMaterial({color:'#2cbdff',transparent:true,opacity:.38,depthWrite:false}))}));
const sparks=Array.from({length:16},()=>ball(scene,glow('#ffcb75'),[0,0,0],.038));const beamLight=new THREE.PointLight('#43caff',0,7,2);scene.add(beamLight);
for(const o of [wrench,leftHand,...arms.flatMap(a=>[a.upper,a.lower,a.hand].filter(Boolean))])o.traverse(m=>{if(m.isMesh){m.castShadow=false;m.receiveShadow=false;}});
let actualTip=V([0,0,0]),impactPoint=null;
function drawBolt(start,end,t){let index=0;for(let branch=0;branch<3;branch++){const points=[];for(let j=0;j<=10;j++){const u=j/10,p=start.clone().lerp(end,u),env=Math.sin(u*Math.PI),tick=Math.floor(t*30);p.x+=env*(Math.sin(j*7.1+tick*2.3+branch)*.10+branch*.035);p.y+=env*Math.cos(j*4.7+tick*1.3+branch)*.075;points.push(p);}for(let j=0;j<10;j++){const o=boltSegments[index++];o.core.visible=o.glow.visible=true;segment(o.core,points[j],points[j+1]);segment(o.glow,points[j],points[j+1]);}}}
function pose(s){
 camera.position.fromArray(s.position);camera.lookAt(...s.target);const age=s.hit?.kind==='wrench'?s.t-s.hit.t:999;if(age<.2)camera.rotateZ(Math.sin(age/.2*Math.PI)*.017);camera.updateMatrixWorld(true);
 for(const e of enemies){const enemy=s.enemies[e.e.id],t=s.t,q={...enemy,health:enemy.hp,stunned:enemy.hp>0&&enemy.stunnedUntil>t,fall:enemy.hp<=0?smooth((t-enemy.deathAt)/.75):0,lastHit:t-enemy.hitAt};e.root.position.set(q.x,.04,q.z);e.root.rotation.y=q.facing;e.body.rotation.set(0,0,q.fall*1.52);e.body.position.set(q.fall*.13,-q.fall*.015,0);if(q.health>0&&q.lastHit<.3&&q.lastKind==='wrench')e.body.rotation.x=Math.sin(q.lastHit/.3*Math.PI)*.23;
 if(q.stunned){e.body.rotation.z+=Math.sin(t*37)*.025;e.body.position.x+=Math.sin(t*43)*.014;}
 for(const a of e.arms){const windup=q.attackAt===null?0:clamp((t-q.attackAt)/.65),lift=q.stunned?.75:windup*.85,shoulder=[a.side*.38,1.3,0],hand=[a.side*(.34+lift*.44),.83+lift*.80,-.15],elbow=[a.side*.57,1.0+lift*.45,-.02];segment(a.upper,shoulder,elbow);segment(a.lower,elbow,hand);a.hand.position.fromArray(hand);}
 e.legs.forEach((leg,i)=>{leg.rotation.x=q.walking?Math.sin(t*9+i*Math.PI)*.25:0;});
 e.arcs.forEach((o,i)=>{o.visible=q.stunned;if(o.visible){const a=i*.77+t*2.0,b=a+.35;segment(o,[q.x+Math.cos(a)*.46,.3+i*.095,q.z+Math.sin(a)*.32],[q.x+Math.cos(b)*.57,.43+i*.095,q.z+Math.sin(b)*.38]);}});
 }
 scene.updateMatrixWorld(true);
 const retract=s.swing?.15:0;leftHand.position.set(-.43+s.ready*.12,-.52+s.ready*.32-retract,-.72-s.ready*.12);leftHand.rotation.set(-s.ready*.20,0,-.12+s.ready*.08);handLight.intensity=s.cast?4: .12;
 let hand=V([.53,-.66,-.56]),head=V([.62,.30,-.78]);impactPoint=null;
 if(s.swing){const contact=s.lastHit?.kind==='wrench'&&Math.abs(s.lastHit.time-s.swing.hit)<.06?s.lastHit:null,targetWorld=contact?V([contact.x,contact.y,contact.z]):camera.localToWorld(V([0,-.48,-2.3])),target=camera.worldToLocal(targetWorld.clone()),impactHand=target.clone().sub(V([-.26,.34,-.904]).normalize().multiplyScalar(1.05)),windHand=V([.56,-.12,-.66]),windHead=V([.69,.80,-.74]);impactPoint=targetWorld;
 if(s.t<s.swing.hit-.28){const u=smooth((s.t-s.swing.start)/.27);hand.lerp(windHand,u);head.lerp(windHead,u);}else if(s.t<=s.swing.hit){const u=smooth((s.t-s.swing.hit+.28)/.28);hand=windHand.lerp(impactHand,u);head=windHead.lerp(target,u);}else {const u=smooth((s.t-s.swing.hit)/.55);hand=impactHand.lerp(hand,u);head=target.lerp(head,u);}
 }
 wrench.position.copy(hand);wrench.quaternion.setFromUnitVectors(up,head.clone().sub(hand).normalize());wrench.updateMatrixWorld(true);actualTip=wrench.localToWorld(V([0,1.05,0]));
 for(const a of arms){const shoulder=[a.side*.34,-.75,.08],end=a.side===1?hand.toArray():leftHand.position.clone().add(V([0,-.025,.1])).toArray(),elbow=shoulder.map((v,i)=>(v+end[i])*.5);elbow[0]+=a.side*.09;elbow[1]-=.06;segment(a.upper,shoulder,elbow);segment(a.lower,elbow,end);if(a.hand)a.hand.position.copy(hand);}
}
function render(state){
 const t=state.time,position=[state.x,state.y+1.68+Math.sin(t*9)*.012*Math.hypot(...state.move),state.z],cp=Math.cos(state.pitch);
 const s={...state,t,position,target:[position[0]+Math.sin(state.yaw)*cp,position[1]+Math.sin(state.pitch),position[2]-Math.cos(state.yaw)*cp],ready:state.cast?1:0,swing:t-state.attackAt<.85?{start:state.attackAt,hit:state.attackAt+.3}:null,hit:state.lastHit&&t-state.lastHit.time<.23?{...state.lastHit,t:state.lastHit.time}:null,water:state.waterUntil>t?1:0};
 pose(s);boltSegments.forEach(o=>o.core.visible=o.glow.visible=false);beamLight.intensity=0;
 if(s.cast){const start=leftHand.localToWorld(V([0,.046,-.29])),end=V([s.cast.end.x,s.cast.end.y,s.cast.end.z]);drawBolt(start,end,s.t);beamLight.position.copy(end).add(V([0,.35,0]));beamLight.intensity=9;}
 poolLight.intensity=s.water*38;water.material.color.set(s.water>.05?'#167f9f':'#185970');water.material.emissive.set('#126884');water.material.emissiveIntensity=s.water*.65;
 rings.forEach((o,i)=>{const u=((s.t*.53+i/3)%1);o.visible=s.water>.01;o.scale.set(POOL.rx*u,POOL.rz*u,1);o.material.opacity=s.water*(1-u)*.75;});
 flow.forEach((o,i)=>{o.visible=s.water>.02;if(o.visible){const a=i*.51,rr=.25+(i%6)*.38,x=POOL.x+Math.cos(a)*rr,z=POOL.z+Math.sin(a)*rr*.68;segment(o,[x,.093,z],[x+Math.sin(i*1.7+s.t*7)*.42,.10,z+Math.cos(i*1.1+s.t*6)*.31]);}});
 sparks.forEach((o,i)=>{o.visible=s.hit?.kind==='wrench';if(o.visible){const e=s.lastHit,age=s.t-s.hit.t,a=i*2.399,r=age*2.8;o.position.set(e.x+Math.cos(a)*r,1.25+Math.sin(a)*r-age*age*1.8,e.z+.2+age*.4);o.scale.setScalar(1-age/.23);}});
 renderer.render(scene,camera);return s;
}
return{render,capture:()=>renderer.domElement.toDataURL('image/png'),captureBlob:()=>new Promise((resolve,reject)=>renderer.domElement.toBlob(b=>b?resolve(b):reject(Error('Unable to capture condition image')),'image/png')),camera,renderer};
}
export function drawHUD(canvas,s,label='LIVE',{gameUI=true,actions=true}={}){
 drawActions(canvas,s,label,{gameUI:false,actions});if(!s)return;const c=canvas.getContext('2d');c.font='16px ui-monospace,monospace';
 if(gameUI){
  c.fillStyle='rgba(8,18,30,.85)';c.fillRect(24,22,360,154);c.fillStyle='#b5e7ef';c.fillText(label,40,47);c.fillStyle='#fff';c.fillText(s.won?'MACHINE ROOM CLEAR':s.dead?'DEFEATED':'ENEMIES '+s.kills+' / 4',40,73);
  c.fillText('HEALTH '+s.health+' / 100',40,100);c.fillStyle='#3e4a53';c.fillRect(40,110,320,7);c.fillStyle=s.health>35?'#84dcca':'#ff8069';c.fillRect(40,110,320*s.health/100,7);
  c.fillStyle='#fff';c.fillText('EVE '+Math.floor(s.eve)+' / 100   E: ELECTRO BOLT',40,141);c.fillStyle='#3e4a53';c.fillRect(40,152,320,7);c.fillStyle='#59c7ff';c.fillRect(40,152,320*s.eve/100,7);
  c.strokeStyle=s.lastHit&&s.time-s.lastHit.time<.18?'#ffcd7e':'#fff';c.lineWidth=2;c.beginPath();c.moveTo(632,352);c.lineTo(648,352);c.moveTo(640,344);c.lineTo(640,360);c.stroke();
  if(s.waterUntil>s.time){c.fillStyle='rgba(8,18,30,.85)';c.fillRect(425,520,430,35);c.fillStyle='#77dfff';c.textAlign='center';c.fillText('WATER ELECTRIFIED — STAY ON DRY FLOOR',640,543);c.textAlign='left';}
  if(s.time-s.hurtAt<.25){c.strokeStyle='#ff8059aa';c.lineWidth=10;c.strokeRect(5,5,1270,694);}
 }
 if(actions){c.fillStyle=s.input.shock?'#7bdfff':'rgba(8,18,30,.85)';c.fillRect(150,621,110,48);c.fillStyle=s.input.shock?'#15232b':'#e7f2f4';c.fillText('E SHOCK',163,651);}
}
