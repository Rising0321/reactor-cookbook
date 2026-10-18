import {combinePrompt} from '../runtime/prompts.js';
export {combinePrompt};
export const FPS=24,DT=1/FPS;
export const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
export const CRATES=[[-2,-1,1.8,1.45,1.8],[4,-10,1.6,1.15,1.5],[7,-19,2,1.55,2],[-4,-13,1.8,1.8,1.8],[13,-28,2,1.6,2],[-4,6,1.4,.85,1.2]];
export const PLANTERS=[[-5,-7],[14,3],[-5,-24],[14,-9]];
export const BLOCKS=[...CRATES.map(([x,z,w,h,d])=>({x,y:h/2,z,w,h,d})),...PLANTERS.map(([x,z])=>({x,y:1.5,z,w:1.4,h:3,d:1.4})),{x:-7,y:4,z:-9,w:2,h:8,d:50},{x:17,y:4,z:-9,w:4,h:8,d:50},{x:4,y:4,z:-32,w:37,h:8,d:3},{x:4,y:4,z:15,w:37,h:8,d:2}];
export const ENEMIES=[[-1.5,0],[6,-2],[1,-12],[10,-15],[1,-23],[11,-25]].map(([x,z],id)=>({id,x,z,wave:Math.floor(id/2)}));
export function segmentBox(a,b,box,pad=0){let lo=0,hi=1;for(const [axis,size] of [['x','w'],['y','h'],['z','d']]){const min=box[axis]-box[size]/2-pad,max=box[axis]+box[size]/2+pad,d=b[axis]-a[axis];if(Math.abs(d)<1e-10){if(a[axis]<min||a[axis]>max)return null;continue;}let p=(min-a[axis])/d,q=(max-a[axis])/d;if(p>q)[p,q]=[q,p];lo=Math.max(lo,p);hi=Math.min(hi,q);if(lo>hi)return null;}return lo;}
export const blocked=(a,b)=>BLOCKS.some(o=>segmentBox(a,b,o)!==null);
const point=(x,y,z)=>({x,y,z});
const historyFrame=s=>({frame:s.frame,x:s.x,y:s.y,z:s.z,yaw:s.yaw,pitch:s.pitch,health:s.health});
export class TracerGame{
 constructor(snapshot){this.s=snapshot?structuredClone(snapshot):{kind:'tracer',frame:0,time:0,x:0,y:0,z:9,yaw:Math.atan2(-1.5,9),pitch:-.04,vy:0,grounded:true,health:100,ammo:24,kills:0,wave:0,nextWaveAt:null,dead:false,won:false,blinkCharges:3,blinkRefillAt:null,blink:null,blinkCount:0,recall:null,recallReadyAt:0,recallCount:0,reloadUntil:0,reloadCount:0,lastShotAt:-10,lastShotHand:0,shotCount:0,hurtAt:-10,lastHit:null,bullets:[],bulletId:0,enemies:ENEMIES.map(e=>({...e,originX:e.x,hp:8,deathAt:null,hitAt:-10,nextShot:1.7+e.id*.4,shotAt:-10})),history:[],events:[],input:{},move:[0,0],look:[0,0]};if(!snapshot)this.s.history.push(historyFrame(this.s));}
 snapshot(){return structuredClone(this.s);}
 living(){return this.s.enemies.filter(e=>e.wave===this.s.wave&&e.hp>0);}
 canStand(x,z){return x>=-5.5&&x<=14.5&&z>=-29&&z<=12.5&&!BLOCKS.some(b=>Math.abs(x-b.x)<b.w/2+.36&&Math.abs(z-b.z)<b.d/2+.36);}
 move(dx,dz){const s=this.s,n=Math.max(1,Math.ceil(Math.hypot(dx,dz)/.12));for(let i=0;i<n;i++){if(this.canStand(s.x+dx/n,s.z))s.x+=dx/n;if(this.canStand(s.x,s.z+dz/n))s.z+=dz/n;}}
 beginReload(){const s=this.s;if(s.ammo===24||s.reloadUntil>s.time)return;s.reloadUntil=s.time+1.15;s.reloadCount++;s.events.push({kind:'reload-start'});}
 emit(owner,from,to,speed){const s=this.s,d=Math.hypot(to.x-from.x,to.y-from.y,to.z-from.z);s.bullets.push({id:s.bulletId++,owner,...from,vx:(to.x-from.x)/d*speed,vy:(to.y-from.y)/d*speed,vz:(to.z-from.z)/d*speed,born:s.time});}
 tickBullets(){const s=this.s,remaining=[];for(const b of s.bullets){const next=point(b.x+b.vx*DT,b.y+b.vy*DT,b.z+b.vz*DT);let first=Infinity,target=null;for(const o of BLOCKS){const hit=segmentBox(b,next,o);if(hit!==null&&hit<first){first=hit;target='wall';}}
   if(b.owner==='player'){for(const e of this.living()){const hit=segmentBox(b,next,{x:e.x,y:1,z:e.z,w:1.1,h:1.95,d:.8},.06);if(hit!==null&&hit<first){first=hit;target=e;}}}else if(!s.recall&&!s.blink){const hit=segmentBox(b,next,{x:s.x,y:s.y+.9,z:s.z,w:.65,h:1.8,d:.65},.04);if(hit!==null&&hit<first){first=hit;target='player';}}
   if(target){if(target==='player'){s.health=Math.max(0,s.health-12);s.hurtAt=s.time;s.events.push({kind:'player-hit',health:s.health});}else if(target!=='wall'){target.hp--;target.hitAt=s.time;s.lastHit={x:target.x,y:1.2,z:target.z,time:s.time};s.events.push({kind:'bullet-hit',target:target.id,hp:target.hp});if(target.hp===0){target.deathAt=s.time;s.kills++;s.events.push({kind:'enemy-killed',target:target.id});}}continue;}
   if(s.time-b.born<(b.owner==='player'?1.4:3))remaining.push({...b,...next});
  }s.bullets=remaining;}
 step(input={}){
  const s=this.s,prev=s.input;s.frame++;s.time=s.frame/FPS;s.events=[];s.move=[0,0];s.look=[0,0];s.input={forward:clamp(input.forward||0,-1,1),side:clamp(input.side||0,-1,1),lookX:clamp(input.lookX||0,-.2,.2),lookY:clamp(input.lookY||0,-.2,.2),fire:!!input.fire,jump:!!input.jump,blink:!!input.blink,recall:!!input.recall,reload:!!input.reload};const u=s.input;if(s.dead||s.won)return this.snapshot();
  if(s.blinkRefillAt!==null&&s.time>=s.blinkRefillAt){s.blinkCharges++;s.blinkRefillAt=s.blinkCharges<3?s.blinkRefillAt+3:null;s.events.push({kind:'blink-recharged',charges:s.blinkCharges});}
  if(s.reloadUntil&&s.time>=s.reloadUntil){s.ammo=24;s.reloadUntil=0;s.events.push({kind:'reload-complete'});}
  if(u.recall&&!prev.recall&&!s.recall&&s.time>=s.recallReadyAt&&s.history.length>1){const path=structuredClone(s.history);s.recall={start:s.time,path,target:path[0]};s.recallReadyAt=s.time+8;s.recallCount++;s.blink=null;s.events.push({kind:'recall-start'});}
  if(s.recall){const r=s.recall,t=clamp((s.time-r.start)/.65),idx=Math.round((1-t)*(r.path.length-1)),p=r.path[idx];for(const k of ['x','y','z','yaw','pitch'])s[k]=p[k];if(t>=1){s.health=r.target.health;s.vy=0;s.grounded=s.y===0;s.recall=null;s.events.push({kind:'recall-complete',health:s.health});}}
  else{
   s.yaw=wrap(s.yaw+u.lookX);s.pitch=clamp(s.pitch+u.lookY,-1.1,.9);s.look=[clamp(u.lookX/.07,-1,1),clamp(-u.lookY/.07,-1,1)];let f=u.forward,r=u.side,len=Math.max(1,Math.hypot(f,r));f/=len;r/=len;s.move=[r,-f];
   if(u.blink&&!prev.blink&&!s.blink&&s.blinkCharges>0){if(!f&&!r)f=1;s.blink={start:s.time,dx:Math.sin(s.yaw)*f+Math.cos(s.yaw)*r,dz:-Math.cos(s.yaw)*f+Math.sin(s.yaw)*r};s.blinkCharges--;s.blinkCount++;s.blinkRefillAt??=s.time+3;s.events.push({kind:'blink-start',forward:f,side:r});}
   if(s.blink){this.move(s.blink.dx*5/.25*DT,s.blink.dz*5/.25*DT);if(s.time-s.blink.start+DT>=.25-1e-8){s.blink=null;s.events.push({kind:'blink-complete'});}}
   else this.move((Math.sin(s.yaw)*f+Math.cos(s.yaw)*r)*4*DT,(-Math.cos(s.yaw)*f+Math.sin(s.yaw)*r)*4*DT);
   if(u.jump&&!prev.jump&&s.grounded){s.vy=4.8;s.grounded=false;s.events.push({kind:'jump'});}if(!s.grounded){s.vy-=12*DT;s.y=Math.max(0,s.y+s.vy*DT);if(s.y===0){s.vy=0;s.grounded=true;}}
   if(u.reload&&!prev.reload)this.beginReload();
   if(u.fire&&!s.blink&&!s.reloadUntil&&s.time-s.lastShotAt>=.125-1e-8){if(s.ammo===0)this.beginReload();else{const hand=s.shotCount%2,side=hand?1:-1,cp=Math.cos(s.pitch),dir=point(Math.sin(s.yaw)*cp,Math.sin(s.pitch),-Math.cos(s.yaw)*cp),from=point(s.x+Math.cos(s.yaw)*side*.32,s.y+1.35,s.z+Math.sin(s.yaw)*side*.32),to=point(s.x+dir.x*25,s.y+1.57+dir.y*25,s.z+dir.z*25);this.emit('player',from,to,34);s.ammo--;s.lastShotAt=s.time;s.lastShotHand=hand;s.shotCount++;s.events.push({kind:'shot',hand,ammo:s.ammo});}}
  }
  for(const e of this.living()){const d=Math.hypot(e.x-s.x,e.z-s.z),from=point(e.x+.4,1.1,e.z),target=point(s.x,s.y+1,s.z);if(d<24&&s.time>=e.nextShot&&!blocked(from,target)){this.emit(e.id,from,target,9);e.nextShot=s.time+1.65;e.shotAt=s.time;s.events.push({kind:'enemy-shot',target:e.id});}}
  this.tickBullets();if(s.health===0){s.dead=true;s.events.push({kind:'player-defeated'});}
  if(!s.dead&&this.living().length===0){if(s.wave===2){s.won=true;s.events.push({kind:'street-cleared'});}else if(s.nextWaveAt===null)s.nextWaveAt=s.time+1.2;else if(s.time>=s.nextWaveAt){s.wave++;s.nextWaveAt=null;for(const e of this.living())e.nextShot=s.time+.8+e.id%2*.5;s.events.push({kind:'wave-start',wave:s.wave+1});}}
  s.history.push(historyFrame(s));s.history=s.history.filter(h=>h.frame>=s.frame-3*FPS);return this.snapshot();
 }
 demoInput(){const s=this.s;if(s.dead||s.won||s.recall)return{};const e=this.living().sort((a,b)=>Math.hypot(a.x-s.x,a.z-s.z)-Math.hypot(b.x-s.x,b.z-s.z))[0];if(!e)return{};
  const dx=e.x-s.x,dz=e.z-s.z,d=Math.hypot(dx,dz),angle=Math.atan2(dx,-dz),delta=wrap(angle-s.yaw),lookX=clamp(delta,-.065,.065),lookY=clamp(Math.atan2(1.5-s.y-1.57,d)-s.pitch,-.04,.04),clear=!blocked(point(s.x,s.y+1.4,s.z),point(e.x,1.5,e.z));
  // Same controls as a player: move along the open central lane before engaging.
  if(!clear||d>13){const wx=Math.abs(s.x)>.25?-s.x:0,wz=wx?0:e.z+(clear?6:3)-s.z,n=Math.max(.1,Math.hypot(wx,wz));return{lookX,lookY,forward:(wx*Math.sin(s.yaw)-wz*Math.cos(s.yaw))/n,side:(wx*Math.cos(s.yaw)+wz*Math.sin(s.yaw))/n};}
  if(s.time>4&&s.recallCount===0)return{recall:true};
  if(s.time>2&&s.blinkCount===0)return{lookX,lookY,side:1,blink:true};
  if(s.time>10&&s.blinkCount===1)return{lookX,lookY,side:-1,blink:true};
  return{lookX,lookY,fire:Math.abs(delta)<.07,reload:s.ammo===0};
 }
}
export const DEFAULTS={first:'Two compact white and orange futuristic pistols with cyan energy cores occupy the lower left and lower right foreground. Red humanoid robots stand on a sunlit street between ochre and teal buildings. Blue cargo crates with orange straps and small trees line the street.',global:'First-person dual-pistol combat in a sunlit stylized city street. Two compact white and orange pulse pistols with glowing cyan cores stay visible in the lower foreground. Red humanoid robots fight among blue cargo crates, ochre and teal buildings and small trees. Maintain weapon identity, street layout and enemy identities. Cyan player projectiles and orange enemy projectiles remain clearly visible. No text or HUD in the scene.',seed:271828};
export function dynamicPrompt(states){const s=states.at(-1);if(!s)return 'Two pulse pistols face red robots waiting ahead on the street.';const events=states.flatMap(s=>s.events||[]),has=k=>events.some(e=>e.kind===k),p=[];
 if(has('shot'))p.push('The twin pistols fire alternately with bright cyan muzzle flashes, recoil and visible cyan projectiles.');if(has('bullet-hit'))p.push('A red robot recoils as the projectiles strike it.');if(has('enemy-killed'))p.push('A defeated robot falls to the pavement and stays down.');if(has('enemy-shot'))p.push('A red robot fires visible orange projectiles toward the player.');if(has('player-hit'))p.push('An enemy projectile hits the player.');
 const blink=events.find(e=>e.kind==='blink-start');if(blink)p.push(`The player makes a short rapid blink ${Math.abs(blink.side)>Math.abs(blink.forward)?blink.side>0?'right':'left':blink.forward>=0?'forward':'backward'}, with cyan speed streaks around the view.`);else if(states.some(s=>s.blink))p.push('The rapid blink continues with cyan speed streaks.');
 if(has('recall-start')||states.some(s=>s.recall))p.push('Cyan temporal trails appear as the viewpoint retraces the recent movement path backward. Defeated robots remain down.');if(has('recall-complete'))p.push('Recall ends at the earlier position and the pistols settle.');if(has('reload-start')||states.some(s=>s.reloadUntil>s.time))p.push('Both pistols tilt outward and lower briefly to reload.');if(has('reload-complete'))p.push('The pistols return to the ready position after reloading.');
 if(states.some(s=>Math.hypot(...s.move)>.05&&!s.recall&&!s.blink))p.push('The player moves through the street.');const yaw=states.reduce((n,s)=>n+(s.input.lookX||0),0),pitch=states.reduce((n,s)=>n+(s.input.lookY||0),0);if(Math.abs(yaw)>.04)p.push(`The view turns ${yaw>0?'right':'left'}.`);if(Math.abs(pitch)>.04)p.push(`The view tilts ${pitch>0?'up':'down'}.`);if(has('wave-start'))p.push('The next pair of robots appears farther along the street.');if(s.won)p.push('All six robots have been defeated. The player stops in the cleared street.');if(s.dead)p.push('The player is defeated and stops.');return p.join(' ')||'The player holds position with the two pistols ready.';
}
