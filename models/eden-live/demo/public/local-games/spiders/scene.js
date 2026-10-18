import * as THREE from 'three';
import {SPIDERS,center,clamp} from './game.js';
import {drawActions} from '../runtime/action-overlay.js';
export function createScene(){
const renderer=new THREE.WebGLRenderer({canvas:document.querySelector('#film'),antialias:true,preserveDrawingBuffer:true});renderer.setSize(1280,704);renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
const scene=new THREE.Scene();scene.background=new THREE.Color('#101d27');scene.fog=new THREE.Fog('#182c36',18,55);
const camera=new THREE.PerspectiveCamera(69,1280/704,.045,90);scene.add(camera);
scene.add(new THREE.HemisphereLight('#a7c9d9','#425358',1.8));const fill=new THREE.DirectionalLight('#96c4e3',1.1);fill.position.set(3,9,8);scene.add(fill);
const lantern=new THREE.PointLight('#ffd08a',9,10,1.8);lantern.position.set(-.45,.1,-.2);camera.add(lantern);
const materials=new Map();function mat(c){if(!materials.has(c))materials.set(c,new THREE.MeshLambertMaterial({color:c,flatShading:true}));return materials.get(c);}
function mesh(g,geo,c,pos=[0,0,0]){const m=new THREE.Mesh(geo,typeof c==='string'?mat(c):c);m.position.set(...pos);m.castShadow=true;m.receiveShadow=true;g.add(m);return m;}
function group(g,pos=[0,0,0]){const a=new THREE.Group();a.position.set(...pos);g.add(a);return a;}
function box(g,c,pos,size){return mesh(g,new THREE.BoxGeometry(...size),c,pos);}
function ball(g,c,pos,scale,detail=1){const m=mesh(g,new THREE.IcosahedronGeometry(1,detail),c,pos);m.scale.set(...scale);return m;}
function rod(g,a,b,r,c){const A=new THREE.Vector3(...a),B=new THREE.Vector3(...b),m=mesh(g,new THREE.CylinderGeometry(r,r,A.distanceTo(B),6),c,A.clone().add(B).multiplyScalar(.5).toArray());m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),B.sub(A).normalize());return m;}
const blockers=[];
box(scene,'#415963',[center(-31),2.6,-31],[13,6,1]);box(scene,'#415963',[center(13),2.6,13],[13,6,1]);
box(scene,'#455259',[0,-.18,-12],[90,.3,110]);
// A continuous faceted tunnel encloses the shot; no platform edge or picture border.
const positions=[],indices=[];const rings=22,sides=12;
for(let j=0;j<rings;j++){const z=15-j*2.65;for(let i=0;i<=sides;i++){const a=i/sides*Math.PI,r=5.9+.28*Math.sin(j*1.8+i*2.4);positions.push(center(z)+Math.cos(a)*r,-.1+Math.sin(a)*(5.6+.2*Math.cos(j*2+i)),z);}}
for(let j=0;j<rings-1;j++)for(let i=0;i<sides;i++){const a=j*(sides+1)+i,b=a+sides+1;indices.push(a,b,a+1,a+1,b,b+1);}
const caveGeo=new THREE.BufferGeometry();caveGeo.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));caveGeo.setIndex(indices);caveGeo.computeVertexNormals();const cave=mesh(scene,caveGeo,new THREE.MeshLambertMaterial({color:'#415963',side:THREE.DoubleSide,flatShading:true}));cave.castShadow=false;blockers.push(cave);
for(let j=0;j<19;j++){
 const z=12-j*2.65;
 for(const side of [-1,1]){
  const m=ball(scene,j%3?'#405661':'#51656a',[center(z)+side*(5.4+(j%2)*.2),1.0+(j%3)*.45,z],[1.3+(j%2)*.3,1.8,1.7],0);m.rotation.set(j*.15,j*.76,j*.13);blockers.push(m);
  const stal=mesh(scene,new THREE.ConeGeometry(.38+(j%3)*.09,1.5+(j%4)*.3,5),'#4c646d',[center(z)+side*3.8,4.1,z-.6]);stal.rotation.z=Math.PI;
 }
 if(j%3===0){const m=ball(scene,'#344b58',[center(z)+2.4,.24,z],[.65,.30,.75],0);blockers.push(m);}
}
const torches=[];
for(let j=0;j<9;j++){
 const z=9-j*5,side=j%2?1:-1,x=center(z)+side*4.5;
 rod(scene,[x,1.5,z],[x-side*.35,2.3,z],.07,'#67462e');box(scene,'#303d42',[x,1.7,z],[.16,.4,.28]);
 const flame=group(scene,[x-side*.35,2.43,z]);const outer=mesh(flame,new THREE.ConeGeometry(.19,.63,5),new THREE.MeshBasicMaterial({color:'#ff982e'}),[0,.16,0]);const inner=mesh(flame,new THREE.ConeGeometry(.09,.4,5),new THREE.MeshBasicMaterial({color:'#ffe28c'}),[0,.13,.03]);outer.castShadow=inner.castShadow=false;
 const light=new THREE.PointLight('#ffa54e',24,11,1.8);light.position.set(x-side*.5,2.5,z);scene.add(light);torches.push({flame,light});
}
for(const z of [-8,-20,-31])for(const side of [-1,1])ball(scene,'#8caaa1',[center(z)+side*4,.5,z],[.45,.7,.45],1);
for(let j=0;j<18;j++){const z=10-j*2.4,x=center(z)+(j%2?3.1:-3.1);const rock=ball(scene,j%3?'#607078':'#738286',[x,.12,z],[.25,.14,.3],0);rock.rotation.y=j;}
const spiders=SPIDERS.map(e=>{
 const root=group(scene),body=group(root);root.scale.setScalar(e.size);
 ball(body,e.id%2?'#704b36':'#725536',[0,.08,.45],[.66,.44,.78]);ball(body,'#47382f',[0,.03,-.35],[.43,.32,.49]);ball(body,'#4b382d',[0,.04,-.76],[.34,.27,.24]);
 // Amber spots and eyes distinguish the creature against blue-grey stone.
 for(const s of [-1,1])ball(body,'#a27643',[s*.35,.35,.44],[.12,.055,.28],0);
 for(let j=0;j<6;j++){const a=(j-2.5)*.085;ball(body,new THREE.MeshBasicMaterial({color:j<2?'#ffaf49':'#ffd185'}),[a,.13+Math.abs(a)*.2,-.972],[.048,.045,.035],1);}
 for(const s of [-1,1]){const fang=mesh(body,new THREE.ConeGeometry(.07,.28,5),'#b9c3a7',[s*.14,-.16,-.89]);fang.rotation.x=-.55;}
 const legs=[];
 for(const side of [-1,1])for(let j=0;j<4;j++){
  const hip=[side*.28,0,-.65+j*.36],knee=[side*(.91+Math.sin(j/3*Math.PI)*.28),.38,-1.03+j*.66],foot=[side*(1.35+Math.sin(j/3*Math.PI)*.3),-.50,-1.45+j*.93];
  const upper=rod(body,hip,knee,.072,'#564334'),lower=rod(body,knee,foot,.049,'#4a382b');const joint=ball(body,'#907046',knee,[.10,.10,.10],0);legs.push({side,j,hip,knee,foot,upper,lower,joint});
 }
 body.traverse(m=>{if(m.isMesh){m.material=m.material.clone();}});return {e,root,body,legs};
});
const axe=group(camera);
rod(axe,[.05,-1.08,.05],[0,.16,0],.047,'#775034');
for(let j=0;j<6;j++){const m=mesh(axe,new THREE.CylinderGeometry(.053,.053,.025,8),'#3d3028',[.034,-.77+j*.065,.035]);m.rotation.z=-.04;}
box(axe,'#526774',[0,.05,0],[.12,.35,.15]);
const bladeShape=new THREE.Shape();bladeShape.moveTo(-.015,.17);bladeShape.lineTo(-.30,.30);bladeShape.lineTo(-.50,.23);bladeShape.lineTo(-.56,.02);bladeShape.lineTo(-.45,-.27);bladeShape.lineTo(-.27,-.23);bladeShape.lineTo(-.02,-.08);bladeShape.closePath();
mesh(axe,new THREE.ExtrudeGeometry(bladeShape,{depth:.075,bevelEnabled:false}),new THREE.MeshStandardMaterial({color:'#9fbfc9',metalness:.48,roughness:.4}),[0,0,-.038]);
const edgeShape=new THREE.Shape();edgeShape.moveTo(-.5,.23);edgeShape.lineTo(-.56,.02);edgeShape.lineTo(-.45,-.27);edgeShape.lineTo(-.40,-.22);edgeShape.lineTo(-.49,.02);edgeShape.lineTo(-.45,.19);edgeShape.closePath();mesh(axe,new THREE.ShapeGeometry(edgeShape),new THREE.MeshBasicMaterial({color:'#e6ece1',side:THREE.DoubleSide}),[0,0,.04]);
box(axe,'#776a4a',[.08,.035,0],[.23,.16,.13]);
axe.traverse(m=>{if(m.isMesh){m.castShadow=false;m.receiveShadow=false;}});
const sparkMat=new THREE.MeshBasicMaterial({color:'#e8cf8a'});const chips=Array.from({length:18},()=>{const m=ball(scene,sparkMat,[0,0,0],[.04,.04,.04],0);m.castShadow=false;return m;});
const slashMaterial=new THREE.MeshBasicMaterial({color:'#d9ede8',transparent:true,opacity:.22,depthWrite:false,side:THREE.DoubleSide});
const slash=mesh(scene,new THREE.TorusGeometry(.45,.014,4,18,Math.PI*.8),slashMaterial);slash.castShadow=false;

const edgePoint=new THREE.Vector3(-.51,.02,0);
function setRod(m,a,b){const A=new THREE.Vector3(...a),B=new THREE.Vector3(...b);m.position.copy(A).add(B).multiplyScalar(.5);m.scale.y=A.distanceTo(B)/m.geometry.parameters.height;m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),B.sub(A).normalize());}
function render(s){
 const t=s.time;camera.position.set(s.x,s.y+1.65,s.z);camera.lookAt(s.x+Math.sin(s.yaw)*Math.cos(s.pitch),s.y+1.65+Math.sin(s.pitch),s.z-Math.cos(s.yaw)*Math.cos(s.pitch));camera.updateMatrixWorld(true);
 spiders.forEach(({root,body,legs},i)=>{const q=s.spiders[i],dead=q.hp<=0?clamp((t-q.deadAt)/.4,0,1):0,bite=q.biteAt===null?0:Math.sin(clamp((t-q.biteAt)/.55,0,1)*Math.PI);
  root.position.set(q.x,.52*q.size-dead*.25,q.z);root.rotation.y=q.facing;body.rotation.z=dead*.4*(i%2?1:-1);body.rotation.x=-bite*.32;body.scale.y=1-dead*.3;
  body.traverse(m=>{if(m.isMesh&&m.material.emissive)m.material.emissive.set(t-q.hitAt<.18?'#a43e17':'#000000');});
  legs.forEach(l=>{const wave=dead?0:Math.sin(t*(q.walking?14:5)+l.j*1.7+l.side)*.13;const knee=[l.knee[0]*(1-.26*dead),l.knee[1]-.13*dead,l.knee[2]+wave];const foot=[l.foot[0]*(1-.48*dead),l.foot[1]*(1-dead)+(-.05)*dead+Math.max(0,wave)*.45,l.foot[2]*(1-.28*dead)+wave];setRod(l.upper,l.hip,knee);setRod(l.lower,knee,foot);l.joint.position.fromArray(knee);});
 });
 const age=t-s.attackAt;
 // Match the cutting edge to the struck creature at impact; misses have a fixed reach.
 const contact=s.lastHit&&Math.abs(s.lastHit.time-s.attackAt-.3)<.07?new THREE.Vector3(s.lastHit.x,s.lastHit.y,s.lastHit.z):camera.localToWorld(new THREE.Vector3(0,-.55,-2.35));
 const local=camera.worldToLocal(contact.clone()).toArray();
 const knots=[[0,[.55,.20,-.95],[-.1,-.12,-.12]],[.17,[.8,.68,-.65],[-.45,.1,-.72]],[.30,local,[.25,.05,.45]],[.43,[-.40,-.55,-1.38],[.6,.05,1.1]],[.72,[.55,.20,-.95],[-.1,-.12,-.12]]];
 let pose=knots[0][1],angles=knots[0][2];if(age>=0&&age<.72){let i=0;while(i<knots.length-2&&age>knots[i+1][0])i++;const a=knots[i],b=knots[i+1],f=clamp((age-a[0])/(b[0]-a[0]),0,1),u=f*f*(3-2*f);pose=a[1].map((n,j)=>n+(b[1][j]-n)*u);angles=a[2].map((n,j)=>n+(b[2][j]-n)*u);}
 axe.rotation.set(...angles);const offset=edgePoint.clone().applyQuaternion(axe.quaternion);axe.position.set(pose[0]-offset.x,pose[1]-offset.y+Math.sin(t*11)*.012*Math.hypot(...s.move),pose[2]-offset.z);axe.updateMatrixWorld(true);
 const h=s.lastHit,hitAge=h?t-h.time:10;chips.forEach((m,j)=>{m.visible=hitAge<.34;if(!m.visible)return;const a=j*2.399,r=hitAge*(1.5+j%3*.5);m.position.set(h.x+Math.cos(a)*r,h.y+hitAge*(1.5+j%2)-hitAge*hitAge*4,h.z+Math.sin(a)*r);m.scale.setScalar(.05*(1-hitAge/.38));});
 slash.visible=hitAge<.13;if(slash.visible){slash.position.set(h.x,h.y+.1,h.z);slash.quaternion.copy(camera.quaternion);slash.rotateZ(-.5);slash.scale.setScalar(1+hitAge*3);}
 torches.forEach(({flame,light},i)=>{flame.scale.y=1+.08*Math.sin(t*7+i);light.intensity=24*(1+.025*Math.sin(t*7+i));});renderer.render(scene,camera);
}
return{render,capture:()=>renderer.domElement.toDataURL('image/png'),captureBlob:()=>new Promise((resolve,reject)=>renderer.domElement.toBlob(b=>b?resolve(b):reject(Error('Unable to capture condition image')),'image/png')),camera,renderer};
}
export function drawHUD(canvas,s,label='LIVE',{gameUI=true,actions=true}={}){
 drawActions(canvas,s,label,{gameUI:false,actions});if(!s||!gameUI)return;const c=canvas.getContext('2d');c.fillStyle='rgba(8,18,30,.82)';c.fillRect(24,22,340,112);c.font='16px ui-monospace, monospace';c.fillStyle='#b5dfe5';c.fillText(label,40,47);c.fillStyle='#fff';c.fillText(s.won?'CAVE CLEARED':s.dead?'DEFEATED':'CLEAR THE CAVE',40,72);c.fillText('HEALTH '+s.health+'   SPIDERS '+s.kills+'/6',40,96);c.fillStyle='#3e4a53';c.fillRect(40,110,300,7);c.fillStyle=s.health>35?'#83dcba':'#ff775b';c.fillRect(40,110,300*s.health/100,7);
 const hit=s.lastHit&&s.time-s.lastHit.time<.18;c.strokeStyle=hit?'#ffc978':'#eefaff';c.lineWidth=2;c.beginPath();for(const a of [-1,1]){c.moveTo(640+a*5,352+a*5);c.lineTo(640+a*11,352+a*11);c.moveTo(640+a*5,352-a*5);c.lineTo(640+a*11,352-a*11);}c.stroke();
 if(s.time-s.hurtAt<.25){c.strokeStyle='rgba(240,74,48,.65)';c.lineWidth=12;c.strokeRect(6,6,1268,692);}
 if(s.spiders.some(e=>e.hp>0&&e.biteAt!==null)){c.fillStyle='#ffc978';c.textAlign='center';c.fillText('BITE INCOMING — STEP BACK OR JUMP',640,535);c.textAlign='left';}
}
