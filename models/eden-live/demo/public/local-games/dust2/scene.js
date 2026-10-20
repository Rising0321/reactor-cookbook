import * as THREE from 'three';
import {ENEMIES,EXIT,clamp} from './game.js';
const WIDTH=1280,HEIGHT=704;
export function createScene(){
const renderer=new THREE.WebGLRenderer({canvas:document.querySelector('#film'),antialias:true,preserveDrawingBuffer:true});renderer.setSize(WIDTH,HEIGHT);renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
const scene=new THREE.Scene();scene.background=new THREE.Color('#54b8df');scene.fog=new THREE.Fog('#dcb678',36,85);
const camera=new THREE.PerspectiveCamera(67,WIDTH/HEIGHT,.045,130);scene.add(camera);
scene.add(new THREE.HemisphereLight('#fff1cc','#90714f',1.7));
const sun=new THREE.DirectionalLight('#ffe9b9',2.35);sun.position.set(-15,26,15);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-31,right:31,top:35,bottom:-35,near:1,far:90});sun.shadow.normalBias=.035;scene.add(sun,sun.target);sun.target.position.set(7,0,-9);
const fill=new THREE.PointLight('#e5f2ff',.35,4);camera.add(fill);
const mats=new Map();function mat(c){if(!mats.has(c))mats.set(c,new THREE.MeshLambertMaterial({color:c,flatShading:true}));return mats.get(c);}
function mesh(g,geo,col,pos=[0,0,0]){const m=new THREE.Mesh(geo,typeof col==='string'?mat(col):col);m.position.set(...pos);m.castShadow=true;m.receiveShadow=true;g.add(m);return m;}
function box(g,col,pos,size){return mesh(g,new THREE.BoxGeometry(...size),col,pos);}
function group(g,pos=[0,0,0]){const a=new THREE.Group();a.position.set(...pos);g.add(a);return a;}
function sphere(g,col,pos,r){return mesh(g,new THREE.IcosahedronGeometry(r,1),col,pos);}
function rod(g,a,b,r,col){const A=new THREE.Vector3(...a),B=new THREE.Vector3(...b),m=mesh(g,new THREE.CylinderGeometry(r,r,A.distanceTo(B),8),col,A.clone().add(B).multiplyScalar(.5).toArray());m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),B.sub(A).normalize());return m;}
const occluders=[];
const wall='#d7a75e',trim='#efc984',shade='#ad7b42';
box(scene,'#d5b47b',[8,-.16,-8],[180,.3,180]);
// Broad volumes establish a sandy alley, arched gate, courtyard and right passage.
function building(x,z,w,d,h,col=wall){occluders.push(box(scene,col,[x,h/2,z],[w,h,d]));box(scene,trim,[x,h+.10,z],[w+.15,.2,d+.15]);return {x,z,w,d,h};}
building(-8,6,4,24,5.8);building(8,6,4,24,6.6);
building(-9,-13,4,14,6.5);building(2,-21,24,4,6.2);
building(10,-5,4,10,5.5);building(10,-20,4,10,5.5);
building(21,-4,20,3,6.4);building(30,-16,3,27,7.5);building(21,-28,21,3,6.0);
// Gate opening has a true hole; the arch stones never overlap a hidden wall face.
for(const x of [-4.65,4.65])building(x,-5,3.3,1.1,5.8);
const archShape=new THREE.Shape();archShape.moveTo(-3,2.5);archShape.absarc(0,2.5,3,Math.PI,0,true);archShape.lineTo(3,5.9);archShape.lineTo(-3,5.9);archShape.closePath();
const archmesh=mesh(scene,new THREE.ExtrudeGeometry(archShape,{depth:1.1,bevelEnabled:false,curveSegments:12}),wall,[0,0,-5.55]);
occluders.push(archmesh);
for(let j=0;j<11;j++){const a=(j+.5)/11*Math.PI;const m=box(scene,trim,[Math.cos(a)*3.05,2.5+Math.sin(a)*3.05,-4.93],[.68,.42,1.22]);m.rotation.z=a-Math.PI/2;}
for(const x of [-3.08,3.08])for(let j=0;j<5;j++)box(scene,j%2?trim:wall,[x,.25+j*.5,-4.93],[.3,.46,1.2]);
function shutter(x,y,z,w=1,h=1.4,turn=0){const g=group(scene,[x,y,z]);g.rotation.y=turn;box(g,'#326c75',[0,0,0],[w,h,.10]);for(let j=0;j<5;j++)box(g,'#235461',[0,-h*.38+j*h*.18,.062],[w*.9,.055,.024]);box(g,trim,[0,-h/2-.09,.04],[w+.2,.14,.17]);}
for(const z of [9,3,-1])shutter(-5.98,3.25,z,1.2,1.6,Math.PI/2);
for(const z of [11,4,-1])shutter(5.98,3.8,z,1.2,1.7,-Math.PI/2);
for(const x of [-5,0,5,10])shutter(x,3.5,-18.97,1.1,1.5);
for(const x of [15,21,26])shutter(x,3.7,-26.46,1.2,1.7);
for(const x of [15,22,27])shutter(x,3.8,-5.54,1.2,1.7,Math.PI);
// Crates are distinct blocks with raised timber braces, avoiding coplanar surfaces.
const SOLIDS=[];
function crate(x,z,w=1.7,h=1.7,d=1.5,y=0){const g=group(scene,[x,y,z]);occluders.push(box(g,'#8e633b',[0,h/2,0],[w,h,d]));for(const Z of [-d/2-.035,d/2+.035]){for(const Y of [.14,h-.14])box(g,'#be8b49',[0,Y,Z],[w+.06,.19,.08]);for(const X of [-w/2+.12,w/2-.12])box(g,'#bc8746',[X,h/2,Z],[.18,h,.08]);const brace=box(g,'#b48143',[0,h/2,Z],[Math.hypot(w*.75,h*.75),.13,.1]);brace.rotation.z=Math.atan2(h,w);}SOLIDS.push({x,z,w,d,h,y});return g;}
crate(-3.9,1,2.3,1.7,2.1);crate(-4.1,1,1.3,1.15,1.3,1.7);crate(4.2,-1,1.6,1.4,2);
crate(8.6,-16,2.2,1.9,1.8);crate(8.5,-16,1.3,1.15,1.2,1.9);crate(-5.7,-17,1.7,1.5,1.5);
crate(24,-8,2.2,1.7,1.8);crate(27.6,-18,1.5,1.8,1.6);crate(20,-25.8,1.7,1.8,1.3);crate(27,-24.5,2.2,1.8,1.8);
function barrel(x,z){mesh(scene,new THREE.CylinderGeometry(.45,.45,1.15,12),'#337987',[x,.575,z]);for(const y of [.15,.95])mesh(scene,new THREE.CylinderGeometry(.466,.466,.1,12),'#2b515d',[x,y,z]);}
barrel(4.7,6);barrel(-5.1,-9.3);barrel(13,-7);barrel(28,-11);
// Door panels, projecting shade cloth and distant rooftop silhouettes.
for(const [x,z,turn] of [[-5.98,6,Math.PI/2],[28.47,-21,-Math.PI/2]]){const g=group(scene,[x,0,z]);g.rotation.y=turn;box(g,'#684f38',[0,1.15,0],[1.7,2.3,.12]);for(const q of [-.4,.4])box(g,'#9b7042',[q,1.15,.072],[.05,2.2,.03]);}
box(scene,'#ac7140',[-3.8,3.55,5],[4.2,.14,2.7]);for(const x of [-5.5,-2.1])rod(scene,[x,2.6,4.1],[x,3.55,6.2],.055,'#76593e');
for(const [x,z] of [[-11,-22],[14,-29],[32,-25]]){box(scene,shade,[x,4.7,z],[3.5,9.4,3.5]);mesh(scene,new THREE.SphereGeometry(1.9,12,8,0,Math.PI*2,0,Math.PI/2),trim,[x,9.4,z]);}
// A few cracks and stones ground the scene without textures or HUD labels.
for(let i=0;i<42;i++){const x=-5+(i*7.31%33),z=13-(i*11.71%40);if((x>5&&x<12&&Math.abs(z+12)>2.5))continue;const m=sphere(scene,i%2?'#b4915b':'#c7a56b',[x,.055,z],.075+(i%3)*.025);m.scale.y=.45;m.castShadow=false;}
function rifle(parent,fp=false){const g=group(parent);box(g,'#38434a',[0,0,-.14],[.13,.15,.43]);box(g,'#925432',[0,-.005,.16],[.12,.15,.24]);box(g,'#b37137',[0,0,-.43],[.13,.13,.24]);rod(g,[0,.025,-.50],[0,.025,-.87],.027,'#29363d');box(g,'#263038',[0,-.17,-.11],[.09,.24,.16]).rotation.x=-.17;box(g,'#282e32',[0,-.12,.065],[.08,.17,.09]).rotation.x=-.25;box(g,'#26343b',[0,.095,-.16],[.055,.035,.16]);box(g,'#2f3d43',[0,.085,-.77],[.04,.12,.035]);box(g,'#243039',[0,.018,-.86],[.065,.065,.065]);g.traverse(m=>{if(m.isMesh&&fp){m.castShadow=false;m.receiveShadow=false;}});return g;}
const weapon=group(camera,[.31,-.31,-.50]);const gun=rifle(weapon,true);
rod(weapon,[.15,-.35,.27],[.05,-.11,.03],.065,'#b79969').castShadow=false;sphere(weapon,'#35433e',[.015,-.11,.03],.078).castShadow=false;
rod(weapon,[-.20,-.37,.18],[-.09,-.08,-.4],.07,'#b79969').castShadow=false;sphere(weapon,'#35433e',[-.065,-.07,-.4],.08).castShadow=false;
const flash=group(weapon,[0,.025,-.91]);
const flashOrange=new THREE.MeshBasicMaterial({color:'#ff991b'}),flashCore=new THREE.MeshBasicMaterial({color:'#fff2a0'});
const flare=mesh(flash,new THREE.ConeGeometry(.105,.40,5),flashOrange,[0,0,-.12]);flare.rotation.x=-Math.PI/2;flare.castShadow=false;
const core=mesh(flash,new THREE.ConeGeometry(.060,.28,5),flashCore,[0,0,-.09]);core.rotation.x=-Math.PI/2;core.castShadow=false;
for(let j=0;j<4;j++){const f=mesh(flash,new THREE.ConeGeometry(.032,.22,3),flashCore,[Math.cos(j*Math.PI/2)*.065,Math.sin(j*Math.PI/2)*.065,-.03]);f.rotation.z=j*Math.PI/2;f.castShadow=false;}
const muzzleLight=new THREE.PointLight('#ffae37',0,4,2);weapon.add(muzzleLight);muzzleLight.position.set(0,0,-.85);
const troops=ENEMIES.map(e=>{
 const root=group(scene),body=group(root);const legs=[];
 for(const s of [-1,1]){const leg=group(body,[s*.17,.65,0]);box(leg,'#58634c',[0,-.28,0],[.24,.56,.25]);box(leg,'#303a35',[0,-.60,-.07],[.27,.14,.38]);legs.push(leg);}
 box(body,'#687b52',[0,1.03,0],[.67,.72,.37]);box(body,'#334840',[0,1.05,-.215],[.57,.5,.13]);
 for(const x of [-.19,0,.19])box(body,'#64705b',[x,1,-.3],[.14,.21,.08]);
 sphere(body,'#cc9a68',[0,1.58,0],.23);const helmet=mesh(body,new THREE.SphereGeometry(.26,10,6,0,Math.PI*2,0,Math.PI*.6),'#435b46',[0,1.65,0]);box(body,'#283e39',[0,1.61,-.205],[.31,.09,.04]);
 rod(body,[-.38,1.32,0],[-.17,1.17,-.4],.11,'#76805d');rod(body,[.37,1.32,0],[.15,1.15,-.28],.11,'#76805d');
 const enemyGun=rifle(body);enemyGun.position.set(.08,1.2,-.24);enemyGun.scale.setScalar(.70);
 const fire=sphere(body,new THREE.MeshBasicMaterial({color:'#ffd767'}),[.08,1.23,-.93],.12);fire.scale.set(1,.6,1.7);fire.castShadow=false;
 return {e,root,body,legs,fire,enemyGun};
});
const tracerMat=new THREE.MeshBasicMaterial({color:'#ffdd69'}),enemyTracerMat=new THREE.MeshBasicMaterial({color:'#ff9d3d'});
const tracers=Array.from({length:12},(_,i)=>{const m=mesh(scene,new THREE.CylinderGeometry(i<7?.018:.025,i<7?.018:.025,1,5),i<7?tracerMat:enemyTracerMat);m.castShadow=false;return m;});
const sparks=Array.from({length:18},()=>{const m=sphere(scene,new THREE.MeshBasicMaterial({color:'#ffe89a'}),[0,0,0],.035);m.castShadow=false;return m;});
const casings=Array.from({length:5},()=>{const m=mesh(scene,new THREE.CylinderGeometry(.018,.018,.07,6),'#dda84d');m.castShadow=false;return m;});
const forward=new THREE.Vector3(0,0,-1),up=new THREE.Vector3(0,1,0),aim=new THREE.Vector3(),localAim=new THREE.Vector3(),muzzle=new THREE.Vector3();
function segment(m,a,b){const A=new THREE.Vector3(...a),B=new THREE.Vector3(...b);m.position.copy(A).add(B).multiplyScalar(.5);m.scale.set(1,A.distanceTo(B),1);m.quaternion.setFromUnitVectors(up,B.sub(A).normalize());m.visible=true;}
const grenade=sphere(scene,'#62785a',[0,0,0],.12);
const flashBurst=sphere(scene,new THREE.MeshBasicMaterial({color:'#fff9df',transparent:true,opacity:0,depthWrite:false}),[0,0,0],1);flashBurst.castShadow=false;
const exitRing=mesh(scene,new THREE.TorusGeometry(EXIT.r,.075,6,48),new THREE.MeshBasicMaterial({color:'#77f1bd'}),[EXIT.x,.05,EXIT.z]);exitRing.rotation.x=-Math.PI/2;exitRing.castShadow=false;
const exitArrow=mesh(scene,new THREE.ConeGeometry(.35,.7,4),new THREE.MeshBasicMaterial({color:'#77f1bd'}),[EXIT.x,1.5,EXIT.z]);exitArrow.rotation.z=Math.PI;exitArrow.castShadow=false;
function render(s){
 const t=s.time,age=t-s.lastShotAt,recoil=age<.15?Math.exp(-age*22)*Math.sin(clamp(age/.035)*Math.PI/2):0,motion=Math.min(1,Math.hypot(...s.move));
 const pitch=s.pitch+recoil*.014,cp=Math.cos(pitch),eyeY=s.y+1.65+(s.dead?-.85:0);
 camera.position.set(s.x,eyeY+Math.sin(t*12)*.015*motion,s.z);camera.lookAt(s.x+Math.sin(s.yaw)*cp,eyeY+Math.sin(pitch),s.z-Math.cos(s.yaw)*cp);camera.updateMatrixWorld(true);
 const reload=s.reloadUntil?Math.sin(clamp(1-(s.reloadUntil-t)/1.8)*Math.PI):0;
 weapon.position.set(.31+Math.sin(t*6)*.012*motion,-.31+Math.sin(t*12)*.012*motion-reload*.18,-.50+recoil*.055);
 const shot=s.trails.at(-1);aim.set(s.x+Math.sin(s.yaw)*30,eyeY+Math.tan(s.pitch)*30,s.z-Math.cos(s.yaw)*30);if(shot&&age<.15)aim.set(shot.end.x,shot.end.y,shot.end.z);
 localAim.copy(aim);camera.worldToLocal(localAim);localAim.sub(weapon.position).normalize();weapon.quaternion.setFromUnitVectors(forward,localAim);weapon.rotateX(recoil*.025);weapon.rotateZ(reload*-.6);weapon.updateMatrixWorld(true);
 flash.visible=age<.065&&!s.reloadUntil;flash.rotation.z=s.shotCount*1.9;muzzleLight.intensity=flash.visible?2.4:0;muzzle.set(0,.025,-.90);weapon.localToWorld(muzzle);
 troops.forEach(({e,root,body,legs,fire,enemyGun})=>{const p=s.enemies[e.id],fall=p.deathAt===null?0:clamp((t-p.deathAt)/.65),stunned=t<p.blindUntil,wind=p.phase==='windup'?clamp(1-(p.fireAt-t)/2):1;
  root.visible=p.active;root.position.set(p.x,.04*fall,p.z);root.rotation.y=p.facing;
  body.position.y=.34*fall;body.rotation.set(stunned?.1:-fall*.12,0,fall*(e.id%2?1:-1)*Math.PI/2+(p.hp>0&&t-p.hitAt<.2?Math.sin(t*48)*.045:0));
  legs.forEach(l=>l.rotation.x=0);enemyGun.rotation.x=stunned?-.55:p.phase==='windup'?.65*(1-wind):p.phase==='idle'?.4:0;fire.visible=p.hp>0&&t-p.shotAt<.065;
 });
 tracers.forEach(m=>m.visible=false);let ti=0;
 for(const tr of s.trails){if(ti>=7)break;const a=t-tr.time,u=clamp(a/.085),from=V(tr.from),end=V(tr.end),A=from.clone().lerp(end,Math.max(0,u-.18)),B=from.clone().lerp(end,u);if(A.distanceTo(B)>.001)segment(tracers[ti++],A.toArray(),B.toArray());}
 ti=7;for(const b of s.bullets.slice(-5)){const q=new THREE.Vector3(b.vx,b.vy,b.vz).normalize().multiplyScalar(.7);segment(tracers[ti++],[b.x-q.x,b.y-q.y,b.z-q.z],[b.x,b.y,b.z]);}
 const hit=s.lastHit,hitAge=hit?t-hit.time:1;sparks.forEach((m,j)=>{m.visible=hitAge<.2;if(!m.visible)return;const a=j*2.399,r=hitAge*(1.5+j%3*.35);m.position.set(hit.x+Math.cos(a)*r,hit.y+Math.sin(a)*r-hitAge*hitAge*3,hit.z+.12);m.scale.setScalar(1-hitAge/.2);});
 casings.forEach((m,j)=>{const tr=s.trails[j];m.visible=!!tr;if(!tr)return;const a=t-tr.time;m.position.set(tr.from.x+Math.cos(s.yaw)*(.2+a*1.4),tr.from.y-.2+a*.8-a*a*2.8,tr.from.z+Math.sin(s.yaw)*(.2+a*1.4));m.rotation.set(a*17,a*11,a*9);});
 grenade.visible=!!s.grenade;if(s.grenade)grenade.position.set(s.grenade.x,s.grenade.y,s.grenade.z);
 const flashAge=t-s.flashAt;flashBurst.visible=!!s.flashPoint&&flashAge<.3;if(flashBurst.visible){flashBurst.position.set(s.flashPoint.x,s.flashPoint.y,s.flashPoint.z);flashBurst.scale.setScalar(.5+flashAge*12);flashBurst.material.opacity=.55*(1-flashAge/.3);}
 exitRing.visible=exitArrow.visible=s.kills===8;if(exitArrow.visible)exitArrow.position.y=1.5+Math.sin(t*3)*.15;
 renderer.render(scene,camera);
}
function V(p){return new THREE.Vector3(p.x,p.y,p.z);}
return {render,capture:()=>renderer.domElement.toDataURL('image/png'),captureBlob:()=>new Promise((resolve,reject)=>renderer.domElement.toBlob(b=>b?resolve(b):reject(Error('Capture failed')),'image/png')),renderer,camera};
}
export function drawHUD(canvas,s,label='LIVE',{gameUI=true,actions=true}={}){
 const c=canvas.getContext('2d');c.clearRect(0,0,canvas.width,canvas.height);if(!s)return;c.font='18px monospace';c.textAlign='left';
 if(gameUI){c.fillStyle='#17221be0';c.fillRect(24,22,400,116);c.fillStyle='#f3e2bd';c.fillText(label,40,47);c.fillStyle='#fff';c.fillText('HEALTH '+s.health+'   AMMO '+s.ammo+' / '+s.reserve,40,78);c.fillText('CLEARED '+s.kills+' / 8   FLASH '+s.flashes,40,105);c.fillStyle='#394741';c.fillRect(40,117,366,7);c.fillStyle=s.health>35?'#8dddba':'#ff9975';c.fillRect(40,117,366*s.health/100,7);
  c.textAlign='center';c.fillStyle='#17221be0';c.fillRect(410,620,460,48);c.fillStyle='#fff';c.fillText(s.won?'EXTRACTION COMPLETE':s.kills===8?'REACH THE GREEN EXTRACTION MARKER':s.kills===2&&s.sector===0?'ADVANCE THROUGH THE ARCH':s.kills===4&&s.sector===1?'TAKE THE RIGHT-HAND PASSAGE':['CLEAR THE ALLEY','CLEAR THE COURTYARD','CLEAR THE STRONGHOLD'][s.sector],640,650);
  if(s.reloadUntil){c.fillStyle='#fff2ba';c.fillText('RELOADING '+Math.max(0,s.reloadUntil-s.time).toFixed(1)+'s',640,440);}
  const aiming=s.enemies.filter(e=>e.active&&e.hp>0&&e.phase==='windup').length;if(aiming){c.fillStyle='#ffbc6c';c.fillText('ENEMY AIMING — MOVE OR TAKE COVER',640,85);}
  c.strokeStyle=s.lastHit&&s.time-s.lastHit.time<.16?'#ffdf80':'#ffffff';c.lineWidth=2;c.beginPath();c.moveTo(630,352);c.lineTo(650,352);c.moveTo(640,342);c.lineTo(640,362);c.stroke();if(s.time-s.hurtAt<.25){c.strokeStyle='#ef755c';c.lineWidth=8;c.strokeRect(4,4,1272,696);}
 }
 if(actions){c.textAlign='center';const u=s.input;[['W',u.forward>0],['A',u.side<0],['S',u.forward<0],['D',u.side>0],['F',u.fire],['R',u.reload],['E',u.flash]].forEach(([key,on],i)=>{const x=24+i*50;c.fillStyle=on?'#e7c58e':'#15211cde';c.fillRect(x,548,44,38);c.fillStyle=on?'#17251e':'#fff';c.fillText(key,x+22,574);});}
}
