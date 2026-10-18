import {combinePrompt} from '../runtime/prompts.js';
export {combinePrompt};
export const FPS=24,DT=1/FPS;
export const clamp=(n,a=0,b=1)=>Math.max(a,Math.min(b,n));
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
export const TARGET_COUNT=6;
export const BUILDINGS=[];
const add=(x,z,target=false)=>{const id=BUILDINGS.length;BUILDINGS.push({id,x,z,w:target?6.5:6,d:target?6.5:6,h:target?12:8+id%4,tone:target?1:id%4,seed:id*7.31,target,maxHp:target?100:60});};
for(const z of [12,-6,-24])for(const x of [-11,11])add(x,z,true);
for(const x of [-32,32])for(const z of [28,10,-8,-26,-42])add(x,z);
for(const x of [-16,0,16])add(x,-43);
export function segmentBox(a,b,box,pad=0){
 let lo=0,hi=1;
 for(const [axis,size] of [['x','w'],['y','h'],['z','d']]){
  const min=box[axis]-box[size]/2-pad,max=box[axis]+box[size]/2+pad,d=b[axis]-a[axis];
  if(Math.abs(d)<1e-10){if(a[axis]<min||a[axis]>max)return null;continue;}
  let p=(min-a[axis])/d,q=(max-a[axis])/d;if(p>q)[p,q]=[q,p];lo=Math.max(lo,p);hi=Math.min(hi,q);if(lo>hi)return null;
 }return lo;
}
const boxOf=b=>({...b,y:b.h/2});
export const distanceToBuilding=(x,z,b)=>Math.hypot(Math.max(0,Math.abs(x-b.x)-b.w/2),Math.max(0,Math.abs(z-b.z)-b.d/2));
const point=(x,y,z)=>({x,y,z});
export function breathRay(s){
 const cp=Math.cos(s.pitch),origin=point(s.x+Math.sin(s.yaw)*3,8.7,s.z-Math.cos(s.yaw)*3);
 const far=point(origin.x+Math.sin(s.yaw)*cp*36,origin.y+Math.sin(s.pitch)*36,origin.z-Math.cos(s.yaw)*cp*36);
 let first=1,target=null;
 for(const b of s.buildings){if(b.hp<=0)continue;const t=segmentBox(origin,far,boxOf(b),.18);if(t!==null&&t<first){first=t;target=b.id;}}
 if(far.y<.1){const floor=(.1-origin.y)/(far.y-origin.y);if(floor<first){first=floor;target=null;}}
 return{origin,end:point(origin.x+(far.x-origin.x)*first,origin.y+(far.y-origin.y)*first,origin.z+(far.z-origin.z)*first),target};
}
export function cameraAt(s){
 const f=[Math.sin(s.yaw),-Math.cos(s.yaw)],r=[Math.cos(s.yaw),Math.sin(s.yaw)];
 const target=point(s.x+f[0]*7,5.5,s.z+f[1]*7),desired=point(s.x-f[0]*29+r[0]*13,26,s.z-f[1]*29+r[1]*13);
 // Pull the camera forward if an intact tower would block the view.
 let reach=1;for(const b of s.buildings){if(b.hp<=0)continue;const hit=segmentBox(target,desired,boxOf(b),.8);if(hit!==null)reach=Math.min(reach,Math.max(.25,hit-.035));}
 return{position:[target.x+(desired.x-target.x)*reach,target.y+(desired.y-target.y)*reach,target.z+(desired.z-target.z)*reach],target:[target.x,target.y,target.z]};
}
export class GodzillaGame{
 constructor(snapshot){this.s=snapshot?structuredClone(snapshot):{
  kind:'godzilla',frame:0,time:0,x:0,y:0,z:30,yaw:Math.atan2(-11,18),pitch:-.08,
  health:100,heat:0,overheated:false,breathing:false,breathStrength:0,breath:null,fireFrames:0,
  stompAt:-10,stomps:0,targets:0,destroyed:0,dead:false,won:false,hurtAt:-10,stride:0,
  buildings:BUILDINGS.map(b=>({...b,hp:b.maxHp,collapseAt:null,burnAt:-10,hitAt:-10,cause:null,nextShot:2.5+b.id*.6,chargeAt:null,aim:null})),
  shells:[],shellId:0,impacts:[],events:[],input:{},move:[0,0],look:[0,0],
 };}
 snapshot(){return structuredClone(this.s);}
 hurtBuilding(b,amount,cause){
  if(b.hp<=0)return;const s=this.s;b.hp=Math.max(0,b.hp-amount);b.hitAt=s.time;if(cause==='fire')b.burnAt=s.time;
  if(b.hp===0){b.collapseAt=s.time;b.cause=cause;b.chargeAt=null;b.aim=null;s.destroyed++;if(b.target)s.targets++;s.events.push({kind:'building-destroyed',target:b.id,defense:b.target,cause});}
 }
 move(dx,dz){
  const s=this.s,n=Math.max(1,Math.ceil(Math.hypot(dx,dz)/.18)),rammed=new Set();
  const allowed=(x,z)=>{if(x< -42||x>42||z< -38||z>38)return false;const hit=s.buildings.find(b=>b.hp>0&&distanceToBuilding(x,z,b)<2.6);if(hit){rammed.add(hit.id);return false;}return true;};
  for(let i=0;i<n;i++){if(allowed(s.x+dx/n,s.z))s.x+=dx/n;if(allowed(s.x,s.z+dz/n))s.z+=dz/n;}
  for(const id of rammed)this.hurtBuilding(s.buildings[id],45*DT,'ram');
  if(rammed.size)s.events.push({kind:'ram',targets:[...rammed]});
 }
 tickShells(){
  const s=this.s,remaining=[];
  for(const p of s.shells){const end=point(p.x+p.vx*DT,p.y+p.vy*DT,p.z+p.vz*DT);let hit=Infinity,target=null;
   for(const b of s.buildings){if(b.hp<=0||b.id===p.owner)continue;const t=segmentBox(p,end,boxOf(b),.2);if(t!==null&&t<hit){hit=t;target='building';}}
   const player=segmentBox(p,end,{x:s.x,y:4.5,z:s.z,w:4.4,h:9,d:4.4},.2);if(player!==null&&player<hit){hit=player;target='player';}
   if(end.y<.2){const floor=(.2-p.y)/(end.y-p.y);if(floor<hit){hit=floor;target='ground';}}
   if(target){s.impacts.push({x:p.x+(end.x-p.x)*hit,y:p.y+(end.y-p.y)*hit,z:p.z+(end.z-p.z)*hit,time:s.time});if(target==='player'){s.health=Math.max(0,s.health-8);s.hurtAt=s.time;s.events.push({kind:'shell-hit',health:s.health});}continue;}
   if(s.time-p.born<4)remaining.push({...p,...end});
  }s.shells=remaining;s.impacts=s.impacts.filter(p=>s.time-p.time<.5);
 }
 step(input={}){
  const s=this.s,prev=s.input;s.frame++;s.time=s.frame/FPS;s.events=[];s.move=[0,0];s.look=[0,0];
  s.input={forward:clamp(input.forward||0,-1,1),side:clamp(input.side||0,-1,1),lookX:clamp(input.lookX||0,-.15,.15),lookY:clamp(input.lookY||0,-.15,.15),fire:!!input.fire,jump:!!input.jump};
  if(s.dead||s.won){s.breathing=false;s.breathStrength=Math.max(0,s.breathStrength-DT*5);return this.snapshot();}const u=s.input;
  s.yaw=wrap(s.yaw+u.lookX);s.pitch=clamp(s.pitch+u.lookY,-.55,.45);s.look=[clamp(u.lookX/.07,-1,1),clamp(-u.lookY/.07,-1,1)];
  const len=Math.max(1,Math.hypot(u.forward,u.side)),f=u.forward/len,r=u.side/len,x=s.x,z=s.z;
  this.move((Math.sin(s.yaw)*f+Math.cos(s.yaw)*r)*7*DT,(-Math.cos(s.yaw)*f+Math.sin(s.yaw)*r)*7*DT);
  const distance=Math.hypot(s.x-x,s.z-z);if(distance>1e-8)s.move=[r,-f];s.stride+=distance*.8;
  if(u.jump&&!prev.jump&&s.time-s.stompAt>=3.5){s.stompAt=s.time;s.stomps++;s.events.push({kind:'stomp'});for(const b of s.buildings)if(b.hp>0&&distanceToBuilding(s.x,s.z,b)<=7.5)this.hurtBuilding(b,65,'stomp');}
  s.breath=breathRay(s);s.breathing=u.fire&&!s.overheated;
  if(s.breathing){s.fireFrames++;s.heat=Math.min(100,s.heat+26*DT);s.breathStrength=Math.min(1,s.breathStrength+DT*5);if(s.breath.target!==null)this.hurtBuilding(s.buildings[s.breath.target],60*DT,'fire');
   if(s.heat>=100){s.overheated=true;s.events.push({kind:'overheated'});}
  }else{s.heat=Math.max(0,s.heat-22*DT);s.breathStrength=Math.max(0,s.breathStrength-DT*5);if(s.overheated&&s.heat<=30){s.overheated=false;s.events.push({kind:'cooled'});}}
  for(const b of s.buildings){if(!b.target||b.hp<=0)continue;
   if(b.chargeAt!==null){if(s.time-b.chargeAt>=.75){const from=point(b.x,b.h+1,b.z),d=Math.hypot(b.aim.x-from.x,b.aim.y-from.y,b.aim.z-from.z);s.shells.push({id:s.shellId++,owner:b.id,...from,vx:(b.aim.x-from.x)/d*18,vy:(b.aim.y-from.y)/d*18,vz:(b.aim.z-from.z)/d*18,born:s.time});s.events.push({kind:'tower-shot',target:b.id});b.chargeAt=null;b.aim=null;b.nextShot=s.time+3.4;}continue;}
   if(Math.hypot(b.x-s.x,b.z-s.z)<42&&s.time>=b.nextShot){b.chargeAt=s.time;b.aim=point(s.x,4.5,s.z);s.events.push({kind:'tower-charge',target:b.id});}
  }
  this.tickShells();
  if(s.health===0){s.dead=true;s.events.push({kind:'monster-defeated'});}
  else if(s.targets===TARGET_COUNT){s.won=true;s.events.push({kind:'city-cleared'});}
  return this.snapshot();
 }
 demoInput(){
  const s=this.s;if(s.dead||s.won)return{};
  const b=s.buildings.filter(b=>b.target&&b.hp>0).sort((a,b)=>Math.hypot(a.x-s.x,a.z-s.z)-Math.hypot(b.x-s.x,b.z-s.z))[0];
  const dx=b.x-s.x,dz=b.z-s.z,d=Math.hypot(dx,dz),delta=wrap(Math.atan2(dx,-dz)-s.yaw),pitch=Math.atan2(b.h*.55-8.7,Math.max(1,d-3));
  const lookX=clamp(delta,-.055,.055),lookY=clamp(pitch-s.pitch,-.035,.035);
  // Advance while cooling, circle slowly while firing; every action is a normal input.
  const approach=s.overheated?7:18,moving=d>approach;
  return{lookX,lookY,forward:moving?Math.cos(delta):0,side:moving?Math.sin(delta):Math.sin(s.time*1.1)*.6,fire:Math.abs(delta)<.10&&d<36,jump:distanceToBuilding(s.x,s.z,b)<7&&s.time-s.stompAt>=3.5};
 }
}
export const DEFAULTS={
 first:'Third-person elevated rear three-quarter view of a huge dark green Godzilla-like reptilian monster standing in a miniature city. Its full body, thick legs, short arms, jagged dorsal plates and long heavy tail are visible. Rectangular grey concrete buildings line broad empty streets. Some taller warm bronze defense towers have rooftop cannons and glowing amber beacons. Overcast daylight, muted blue-grey sky, no people, text or UI.',
 global:'Third-person elevated following camera behind exactly one giant dark green reptilian kaiju rampaging through a miniature city. There is only one monster in the entire scene, always the same continuous body with jagged dorsal plates and one long tail. Keep the same city layout. The monster breathes a continuous orange-yellow flame jet from its open mouth, burning buildings into chunks and dust. Stomps create ground shockwaves. Warm bronze defense towers fire slow visible orange cannon shells. Collapsed buildings remain as low piles of grey and brown angular concrete blocks and dust. Rubble has no scales, limbs, heads or tails. No additional monsters, duplicate bodies or monster-shaped debris appear. Destroyed buildings never reappear. No text, HUD or human figures.',seed:68427,
};
export function dynamicPrompt(states){
 const s=states.at(-1);if(!s)return 'The giant reptilian monster stands among the city streets with its mouth closed.';
 const events=states.flatMap(s=>s.events||[]),has=k=>events.some(e=>e.kind===k),p=[];
 if(states.some(s=>s.breathing))p.push('The monster opens its jaws and breathes a continuous orange-yellow flame jet toward the aimed point.');
 else p.push('The monster stops breathing fire and closes its jaws.');
 if(events.some(e=>e.kind==='building-destroyed'&&e.cause==='fire'))p.push('A burning building collapses into concrete chunks and a rising dust cloud.');
 if(has('stomp')||states.some(s=>s.time-s.stompAt<.6))p.push('The monster stamps a heavy foot onto the road, sending a circular ground shockwave into nearby buildings.');
 if(has('ram'))p.push('The monster pushes its body against a nearby building, cracking the structure.');
 if(events.some(e=>e.kind==='building-destroyed'&&e.cause!=='fire'))p.push('A struck building crumbles into rubble around the monster.');
 if(has('tower-shot')||states.some(s=>s.shells.length))p.push('Rooftop defense cannons launch visible orange shells toward the monster.');
 if(has('shell-hit'))p.push('An orange shell hits the monster with a brief burst of sparks.');
 if(has('overheated')||s.overheated)p.push('The flame jet shuts off while the monster cools down.');
 if(states.some(s=>Math.hypot(...s.move)>.05))p.push('The monster walks with heavy alternating steps and a swaying tail; the camera follows from above and behind.');
 const turn=states.reduce((n,s)=>n+(s.input.lookX||0),0),tilt=states.reduce((n,s)=>n+(s.input.lookY||0),0);
 if(Math.abs(turn)>.04)p.push(`The monster turns ${turn>0?'right':'left'} and the following camera rotates with it.`);if(Math.abs(tilt)>.04)p.push(`Its head aims ${tilt>0?'higher':'lower'}.`);
 if(s.won)p.push('All six defense towers are destroyed. The monster stands victorious among the ruins.');if(s.dead)p.push('The monster is defeated, slumps down and stops moving.');
 return p.join(' ');
}
