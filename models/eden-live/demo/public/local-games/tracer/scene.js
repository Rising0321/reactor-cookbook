import * as THREE from 'three';
import {ENEMIES,CRATES,PLANTERS,clamp} from './game.js';
import {drawActions} from '../runtime/action-overlay.js';
export function createScene(){
const renderer=new THREE.WebGLRenderer({canvas:document.querySelector('#film'),antialias:true,preserveDrawingBuffer:true});
renderer.setSize(1280,704);renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
const scene=new THREE.Scene();scene.background=new THREE.Color('#319ed6');scene.fog=new THREE.Fog('#a3d7e2',48,100);
const camera=new THREE.PerspectiveCamera(74,1280/704,.04,130);scene.add(camera);
scene.add(new THREE.HemisphereLight('#fff4dd','#344b68',2));const sun=new THREE.DirectionalLight('#fff0d5',2.6);sun.position.set(-12,26,17);sun.target.position.set(4,0,-9);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-28,right:28,top:33,bottom:-33,near:1,far:80});sun.shadow.normalBias=.035;scene.add(sun,sun.target);
const materials=new Map();function mat(c){if(!materials.has(c))materials.set(c,new THREE.MeshLambertMaterial({color:c,flatShading:true}));return materials.get(c);}
function mesh(g,geo,c,p=[0,0,0]){const m=new THREE.Mesh(geo,typeof c==='string'?mat(c):c);m.position.set(...p);m.castShadow=true;m.receiveShadow=true;g.add(m);return m;}
function group(g,p=[0,0,0]){const a=new THREE.Group();a.position.set(...p);g.add(a);return a;}
function box(g,c,p,s){return mesh(g,new THREE.BoxGeometry(...s),c,p);}
function ball(g,c,p,r=.1){return mesh(g,new THREE.IcosahedronGeometry(r,1),c,p);}
function rod(g,a,b,r,c){const A=new THREE.Vector3(...a),B=new THREE.Vector3(...b),m=mesh(g,new THREE.CylinderGeometry(r,r,A.distanceTo(B),8),c,A.clone().add(B).multiplyScalar(.5).toArray());m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),B.sub(A).normalize());return m;}
const cyan=new THREE.MeshBasicMaterial({color:'#31e7ff'}),white=new THREE.MeshBasicMaterial({color:'#f1ffff'}),orange=new THREE.MeshBasicMaterial({color:'#ffad26'});
const occluders=[];function solid(c,p,s){const m=box(scene,c,p,s);occluders.push(m);return m;}
box(scene,'#809fa7',[4,-.12,-8],[70,.24,100]);
// Coarse painted volumes and large shadow shapes, with no textures or HUD.
for(const side of [-1,1])for(let i=0;i<6;i++){
 const x=side<0?-10:19,z=12-i*8,h=4.5+(i%3)*1.2,w=7,d=7.9;
 solid(['#e6c695','#239797','#ec9b44'][(i+(side>0?1:0))%3],[x,h/2,z],[w,h,d]);
 box(scene,'#e9dfbe',[x,h+.1,z],[w+.22,.2,d+.2]);
 const face=x+(side<0?1:-1)*(w/2+.04);
 for(const zz of [z-2,z+1.6]){box(scene,'#18334d',[face,2.9,zz],[.12,1.2,1.25]);box(scene,'#dfd4a6',[face+(side<0?.09:-.09),2.24,zz],[.28,.13,1.47]);}
 if(i%2===0){box(scene,'#19435a',[face,1.1,z],[.13,2.2,1.3]);box(scene,'#ee7932',[face+(side<0?1:-1)*.6,2.6,z],[1.6,.14,2.2]);}
}
solid('#ddbe87',[4,3,-32],[37,6,3]);for(const x of [-5,1,7,13])box(scene,'#285672',[x,3.6,-30.43],[1.5,1.6,.13]);
for(const side of [-1,1])box(scene,'#bad0c4',[side<0?-6:15,.06,-8],[.7,.12,47]);
for(let i=0;i<10;i++)box(scene,'#e6d8ae',[4,.008,11-i*4.3],[.18,.016,1.3]);
function crate(x,z,w=1.8,h=1.45,d=1.8){solid('#35677b',[x,h/2,z],[w,h,d]);for(const zz of [z-d/2-.04,z+d/2+.04]){for(const xx of [x-w*.32,x+w*.32])box(scene,'#f2ad36',[xx,h/2,zz],[.12,h+.02,.07]);box(scene,'#173e56',[x,h*.7,zz],[w*.42,.2,.08]);}}
for(const c of CRATES)crate(...c);
// Place foliage outside the action lanes, as crude faceted clumps.
for(const [x,z] of PLANTERS){solid('#b87c4e',[x,.3,z],[1.15,.6,1.15]);rod(scene,[x,.5,z],[x,2.1,z],.13,'#735335');const crown=ball(scene,'#278461',[x,2.3,z],1);crown.scale.set(1,1.15,1);}
const forward=new THREE.Vector3(0,0,-1),up=new THREE.Vector3(0,1,0);
const guns=[-1,1].map(side=>{
 const root=group(camera);
 box(root,'#243b50',[0,-.13,-.02],[.115,.26,.14]).rotation.x=-.18;
 box(root,'#e9eee8',[0,.025,-.23],[.25,.24,.53]);box(root,'#f68a14',[side*.07,.06,-.23],[.13,.20,.47]);
 box(root,'#183b50',[0,.03,-.54],[.19,.16,.14]);box(root,cyan,[0,.035,-.615],[.135,.074,.013]);
 const core=mesh(root,new THREE.CylinderGeometry(.104,.104,.275,12),cyan,[0,.08,-.22]);core.rotation.z=Math.PI/2;
 for(const x of [-.153,.153])mesh(root,new THREE.CylinderGeometry(.11,.11,.028,10),'#e5edeb',[x,.08,-.22]).rotation.z=Math.PI/2;
 box(root,'#263b50',[0,.16,-.30],[.055,.03,.22]);box(root,cyan,[0,.18,-.31],[.03,.018,.12]);
 const flash=group(root,[0,.035,-.66]);const flare=mesh(flash,new THREE.ConeGeometry(.14,.43,5),cyan,[0,0,-.16]);flare.rotation.x=-Math.PI/2;const center=mesh(flash,new THREE.ConeGeometry(.065,.32,5),white,[0,0,-.12]);center.rotation.x=-Math.PI/2;
 root.traverse(o=>{if(o.isMesh){o.castShadow=false;o.receiveShadow=false;}});return{root,core,flash,side};
});
const robots=ENEMIES.map(e=>{const root=group(scene,[e.x,0,e.z]),body=group(root);
 for(const side of [-1,1]){box(body,'#283e54',[side*.21,.4,0],[.27,.68,.31]);box(body,'#ec5146',[side*.21,.13,-.04],[.31,.24,.44]);box(body,'#ab2b3a',[side*.48,1.12,0],[.25,.61,.30]);box(body,'#283e54',[side*.48,.91,-.21],[.23,.22,.48]);}
 box(body,'#d83c46',[0,1.10,0],[.75,.72,.46]);box(body,'#ff9d53',[0,1.13,-.252],[.30,.24,.06]);box(body,'#df4147',[0,1.69,0],[.48,.44,.43]);box(body,'#132d46',[0,1.69,-.231],[.36,.13,.03]);box(body,orange,[0,1.70,-.253],[.29,.045,.025]);
 const fire=ball(body,orange,[.48,.94,-.5],.15);fire.scale.z=1.7;
 body.traverse(m=>{if(m.isMesh)m.material=m.material.clone();});return{e,root,body,fire};});

const bolts=Array.from({length:80},()=>{const m=mesh(scene,new THREE.CylinderGeometry(.065,.065,1,6),cyan);m.castShadow=false;return m;});
function segment(m,a,b){const d=b.clone().sub(a);m.position.copy(a).add(b).multiplyScalar(.5);m.scale.set(1,Math.max(.001,d.length()),1);m.quaternion.setFromUnitVectors(up,d.normalize());}
const sparks=Array.from({length:16},(_,j)=>{const m=ball(scene,j%3?cyan:white,[0,0,0],.06);m.castShadow=false;return m;});
const streaks=Array.from({length:20},(_,j)=>{const m=box(camera,j%3?cyan:white,[0,0,0],[.018,.018,1]);m.castShadow=false;return m;});
const ghosts=Array.from({length:12},()=>{const m=ball(scene,new THREE.MeshBasicMaterial({color:'#39dcff',transparent:true,opacity:.4,depthWrite:false}),[0,0,0],.14);m.castShadow=false;return m;});
function render(s){const t=s.time;camera.position.set(s.x,s.y+1.57,s.z);camera.lookAt(s.x+Math.sin(s.yaw)*Math.cos(s.pitch),s.y+1.57+Math.sin(s.pitch),s.z-Math.cos(s.yaw)*Math.cos(s.pitch));camera.updateMatrixWorld(true);
 const recalling=!!s.recall,reload=s.reloadUntil>t?Math.sin(Math.PI*clamp((1.15-(s.reloadUntil-t))/1.15)):0;
 guns.forEach((g,i)=>{const age=s.lastShotHand===i?t-s.lastShotAt:10,recoil=age<.12?Math.sin(Math.PI*age/.12):0;g.root.position.set(g.side*(.41+reload*.14),-.32-reload*.13-(recalling?.12:0),-.70+recoil*.05);g.root.quaternion.setFromUnitVectors(forward,new THREE.Vector3(-g.side*.41,.03,-14).normalize());g.root.rotateX(recoil*.06+reload*.5);g.root.rotateZ(g.side*(reload*1.3+(recalling?.4:0)));g.flash.visible=age<.09;g.flash.rotation.z=Math.floor(t*31);});
 robots.forEach(({root,body,fire},i)=>{const e=s.enemies[i];root.visible=e.wave<=s.wave;root.position.set(e.x,0,e.z);if(e.hp>0)root.rotation.y=Math.atan2(e.x-s.x,e.z-s.z);const fall=e.hp<=0?clamp((t-e.deathAt)/.65):0;body.rotation.x=-fall*Math.PI/2;body.position.y=.24*fall;body.position.z=-fall*.35;body.rotation.z=e.hp>0&&t-e.hitAt<.15?.07*Math.sin(t*75):0;body.traverse(m=>{if(m.isMesh&&m.material.emissive)m.material.emissive.set(t-e.hitAt<.13?'#803d08':'#000000');});fire.visible=e.hp>0&&t-e.shotAt<.1;});
 bolts.forEach((m,i)=>{const b=s.bullets[i];m.visible=!!b;if(!b)return;m.material=b.owner==='player'?cyan:orange;const end=new THREE.Vector3(b.x,b.y,b.z),v=new THREE.Vector3(b.vx,b.vy,b.vz).normalize().multiplyScalar(b.owner==='player'?.65:.8);segment(m,end.clone().sub(v),end);});
 const hit=s.lastHit,age=hit?t-hit.time:10;sparks.forEach((m,j)=>{m.visible=age<.22;if(!m.visible)return;const a=j*2.399,r=age*(1.6+j%4*.3);m.position.set(hit.x+Math.cos(a)*r,hit.y+Math.sin(a)*r,hit.z+.15);m.scale.setScalar(1-age/.24);});
 streaks.forEach((m,j)=>{m.visible=!!s.blink||recalling;if(!m.visible)return;const a=j/20*Math.PI*2,z=-1.2-((j*.71+t*8)%3);m.position.set(Math.cos(a)*(-z)*1.2,Math.sin(a)*(-z)*.69,z);m.scale.z=s.blink?1.3:.6;});
 ghosts.forEach((m,j)=>{m.visible=recalling;if(!recalling)return;const p=s.recall.path[Math.floor(j/12*(s.recall.path.length-1))];m.position.set(p.x+Math.cos(j*2.4)*.3,p.y+.8+(j%3)*.25,p.z);});renderer.render(scene,camera);
}
return{render,capture:()=>renderer.domElement.toDataURL('image/png'),captureBlob:()=>new Promise((resolve,reject)=>renderer.domElement.toBlob(b=>b?resolve(b):reject(Error('Unable to capture condition image')),'image/png')),camera,renderer};
}
export function drawHUD(canvas,s,label='LIVE',{gameUI=true,actions=true}={}){
 drawActions(canvas,s,label,{gameUI:false,actions});if(!s)return;const c=canvas.getContext('2d');c.font='16px ui-monospace,monospace';
 if(gameUI){c.fillStyle='rgba(8,18,30,.82)';c.fillRect(24,22,360,112);c.fillStyle='#b5e7ef';c.fillText(label,40,47);c.fillStyle='#fff';c.fillText(s.won?'STREET CLEARED':s.dead?'DEFEATED':'ROBOTS '+s.kills+'/6   WAVE '+(s.wave+1)+'/3',40,74);c.fillText('HEALTH '+s.health+'    AMMO '+s.ammo+'/24',40,100);c.fillStyle='#3e4a53';c.fillRect(40,113,320,7);c.fillStyle=s.health>35?'#84dcca':'#ff8069';c.fillRect(40,113,320*s.health/100,7);
 c.fillStyle='rgba(8,18,30,.82)';c.fillRect(380,601,470,80);c.fillStyle='#fff';c.fillText('SHIFT  BLINK',400,629);for(let i=0;i<3;i++){c.fillStyle=i<s.blinkCharges?'#73f0f3':'#405264';c.fillRect(405+i*30,643,22,11);}c.fillStyle='#b5e7ef';c.fillText(s.blinkRefillAt?Math.max(0,s.blinkRefillAt-s.time).toFixed(1)+'s':'READY',507,655);c.fillText('E  RECALL',625,629);c.fillText(s.recall?'REWINDING':s.recallReadyAt>s.time?(s.recallReadyAt-s.time).toFixed(1)+'s':'READY',625,655);
 c.strokeStyle=s.lastHit&&s.time-s.lastHit.time<.15?'#ffbd61':'#fff';c.lineWidth=2;c.beginPath();c.moveTo(632,352);c.lineTo(648,352);c.moveTo(640,344);c.lineTo(640,360);c.stroke();if(s.reloadUntil>s.time){c.fillStyle='#fff';c.textAlign='center';c.fillText('RELOADING',640,405);c.textAlign='left';}if(s.time-s.hurtAt<.2){c.strokeStyle='#ff8059aa';c.lineWidth=10;c.strokeRect(5,5,1270,694);}}
 if(actions){for(const [i,key,on] of [[0,'SHIFT',s.input.blink],[1,'E',s.input.recall],[2,'R',s.input.reload]]){const x=150+i*72;c.fillStyle=on?'#ecb477':'rgba(8,18,30,.76)';c.fillRect(x,621,64,48);c.fillStyle=on?'#15232b':'#e7f2f4';c.fillText(key,x+8,651);}}
}
