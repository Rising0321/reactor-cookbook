import {combinePrompt} from '../runtime/prompts.js';
export {combinePrompt};
export const FPS=24,DT=1/FPS;
export const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
const point=(x,y,z)=>({x,y,z});
export const BUILDINGS=[[-8,6,4,24,5.8],[8,6,4,24,6.6],[-9,-13,4,14,6.5],[2,-21,24,4,6.2],[10,-5,4,10,5.5],[10,-20,4,10,5.5],[21,-4,20,3,6.4],[30,-16,3,27,7.5],[21,-28,21,3,6],[-4.65,-5,3.3,1.1,5.8],[4.65,-5,3.3,1.1,5.8]];
export const CRATES=[[-3.9,1,2.3,1.7,2.1,0],[-4.1,1,1.3,1.15,1.3,1.7],[4.2,-1,1.6,1.4,2,0],[8.6,-16,2.2,1.9,1.8,0],[8.5,-16,1.3,1.15,1.2,1.9],[-5.7,-17,1.7,1.5,1.5,0],[24,-8,2.2,1.7,1.8,0],[27.6,-18,1.5,1.8,1.6,0],[20,-25.8,1.7,1.8,1.3,0],[27,-24.5,2.2,1.8,1.8,0]];
export const BARRELS=[[4.7,6],[-5.1,-9.3],[13,-7],[28,-11]];
export const BLOCKS=[...BUILDINGS.map(([x,z,w,d,h])=>({x,y:h/2,z,w,h,d})),...CRATES.map(([x,z,w,h,d,y])=>({x,y:y+h/2,z,w,h,d})),...BARRELS.map(([x,z])=>({x,y:.575,z,w:.9,h:1.15,d:.9}))];
// Narrow slabs approximate the curved solid arch above its open passage.
for(let i=0;i<24;i++){const x=-3+(i+.5)*.25,bottom=2.5+Math.sqrt(9-x*x);BLOCKS.push({x,y:(bottom+5.9)/2,z:-5,w:.25,h:5.9-bottom,d:1.1});}
export const ENEMIES=[[-1.6,-.5,0],[2.2,-2.8,0],[6,-15,1],[-4,-16,1],[21,-10,2],[23,-20,2],[26,-15,3],[22,-24,3]].map(([x,z,wave],id)=>({id,x,z,wave}));
export const EXIT={x:22,z:-24.2,r:1.5};
export const ROUTE=[[0,6],[0,-8],[2,-11],[6.5,-12],[13,-12],[19,-13],[22,-22],[EXIT.x,EXIT.z]];
export function segmentBox(a,b,box,pad=0){let lo=0,hi=1;for(const [axis,size] of [['x','w'],['y','h'],['z','d']]){const min=box[axis]-box[size]/2-pad,max=box[axis]+box[size]/2+pad,d=b[axis]-a[axis];if(Math.abs(d)<1e-10){if(a[axis]<min||a[axis]>max)return null;continue;}let p=(min-a[axis])/d,q=(max-a[axis])/d;if(p>q)[p,q]=[q,p];lo=Math.max(lo,p);hi=Math.min(hi,q);if(lo>hi)return null;}return lo;}
export const blocked=(a,b)=>BLOCKS.some(o=>segmentBox(a,b,o)!==null);
export class Dust2Game{
 constructor(snapshot){this.s=snapshot?structuredClone(snapshot):{kind:'dust2',frame:0,time:0,x:0,y:0,z:14,yaw:0,pitch:-.04,vy:0,health:100,ammo:30,reserve:90,flashes:2,flashCount:0,flashAt:-10,flashPoint:null,grenade:null,kills:0,sector:0,nextWaveAt:null,route:0,exitProgress:0,dead:false,won:false,reloadUntil:0,reloadCount:0,lastShotAt:-10,shotCount:0,hurtAt:-10,lastHit:null,trails:[],bullets:[],bulletId:0,enemies:ENEMIES.map(e=>({...e,hp:100,active:e.wave===0,spawnAt:e.wave===0?0:null,deathAt:null,hitAt:-10,phase:'idle',aim:null,fireAt:0,burstLeft:0,nextShot:e.id*.35,blindUntil:0,shotAt:-10,facing:Math.atan2(e.x,e.z-14)})),events:[],input:{},move:[0,0],look:[0,0]};}
 snapshot(){return structuredClone(this.s);}
 living(){return this.s.enemies.filter(e=>e.active&&e.hp>0);}
 canStand(x,z){return x>=-6.5&&x<=28&&z>=-26&&z<=17&&!BLOCKS.some(b=>b.y-b.h/2<1.8&&b.y+b.h/2>0&&Math.abs(x-b.x)<b.w/2+.32&&Math.abs(z-b.z)<b.d/2+.32)&&!this.living().some(e=>Math.hypot(x-e.x,z-e.z)<.62);}
 move(dx,dz){const s=this.s,n=Math.max(1,Math.ceil(Math.hypot(dx,dz)/.1));for(let i=0;i<n;i++){if(this.canStand(s.x+dx/n,s.z))s.x+=dx/n;if(this.canStand(s.x,s.z+dz/n))s.z+=dz/n;}}
 aimTarget(){const s=this.s,from=point(s.x,s.y+1.65,s.z),cp=Math.cos(s.pitch),dir=point(Math.sin(s.yaw)*cp,Math.sin(s.pitch),-Math.cos(s.yaw)*cp);return this.living().map(e=>{const end=point(e.x,1.3,e.z),d=Math.hypot(end.x-from.x,end.y-from.y,end.z-from.z),dot=((end.x-from.x)*dir.x+(end.y-from.y)*dir.y+(end.z-from.z)*dir.z)/d;return{e,end,d,dot};}).filter(t=>t.d<=34&&t.dot>=Math.cos(.085)&&!blocked(from,t.end)).sort((a,b)=>b.dot-a.dot||a.d-b.d)[0]||null;}
 reload(){const s=this.s;if(s.reloadUntil||s.ammo===30||s.reserve===0)return;s.reloadUntil=s.time+1.8;s.reloadCount++;s.events.push({kind:'reload'});}
 shoot(){
  const s=this.s,from=point(s.x,s.y+1.65,s.z),assist=this.aimTarget(),cp=Math.cos(s.pitch);let end=assist?assist.end:point(from.x+Math.sin(s.yaw)*cp*40,from.y+Math.sin(s.pitch)*40,from.z-Math.cos(s.yaw)*cp*40),limit=1,target=null;
  for(const b of BLOCKS){const h=segmentBox(from,end,b);if(h!==null&&h<limit){limit=h;target=null;}}
  for(const e of this.living()){const h=segmentBox(from,end,{x:e.x,y:.95,z:e.z,w:.7,h:1.9,d:.65});if(h!==null&&h<limit){limit=h;target=e;}}
  end=point(from.x+(end.x-from.x)*limit,from.y+(end.y-from.y)*limit,from.z+(end.z-from.z)*limit);
  s.ammo--;s.shotCount++;s.lastShotAt=s.time;s.trails.push({from,end,time:s.time});s.events.push({kind:'shot'});
  if(target){target.hp=Math.max(0,target.hp-34);target.hitAt=s.time;s.lastHit={...end,time:s.time};s.events.push({kind:'hit',enemy:target.id});if(target.hp===0){target.deathAt=s.time;target.phase='idle';target.aim=null;s.kills++;s.events.push({kind:'kill',enemy:target.id});}}
 }
 throwFlash(){const s=this.s;if(!s.flashes||s.grenade)return;s.flashes--;s.flashCount++;s.grenade={x:s.x,y:s.y+1.4,z:s.z,vx:Math.sin(s.yaw)*10,vy:3,vz:-Math.cos(s.yaw)*10,born:s.time};s.events.push({kind:'flash-throw'});}
 flashTick(){const s=this.s,g=s.grenade;if(!g)return;const next=point(g.x+g.vx*DT,Math.max(.15,g.y+g.vy*DT),g.z+g.vz*DT);if(blocked(g,next)){g.vx=g.vz=0;}else Object.assign(g,next);g.vy-=10*DT;
  if(s.time-g.born>=.65){s.flashAt=s.time;s.flashPoint=point(g.x,g.y,g.z);for(const e of this.living())if(Math.hypot(e.x-g.x,e.z-g.z)<10&&!blocked(g,point(e.x,1.55,e.z))){e.blindUntil=s.time+4;e.phase='idle';e.aim=null;e.nextShot=e.blindUntil+1;s.events.push({kind:'blinded',enemy:e.id});}s.grenade=null;s.events.push({kind:'flash-burst'});}
 }
 enemyTick(){const s=this.s;let engaged=this.living().filter(e=>e.phase!=='idle').length;
  for(const e of this.living()){
   const from=point(e.x,1.25,e.z),target=point(s.x,s.y+1.05,s.z),visible=Math.hypot(e.x-s.x,e.z-s.z)<27&&!blocked(from,target);
   if(!visible||s.time<e.blindUntil){if(e.phase!=='idle'){e.phase='idle';e.aim=null;e.nextShot=s.time+1;engaged--;}continue;}
   if(e.phase==='idle'&&engaged<2&&s.time>=e.nextShot){e.phase='windup';e.aim={...target};e.fireAt=s.time+2;e.burstLeft=3;e.facing=Math.atan2(e.x-s.x,e.z-s.z);engaged++;s.events.push({kind:'enemy-windup',enemy:e.id});}
   if(e.phase!=='idle'&&s.time>=e.fireAt){e.phase='burst';const d=Math.hypot(e.aim.x-from.x,e.aim.y-from.y,e.aim.z-from.z);s.bullets.push({id:s.bulletId++,...from,vx:(e.aim.x-from.x)/d*13,vy:(e.aim.y-from.y)/d*13,vz:(e.aim.z-from.z)/d*13,born:s.time});e.shotAt=s.time;e.fireAt=s.time+.28;e.burstLeft--;s.events.push({kind:'enemy-shot',enemy:e.id});if(!e.burstLeft){e.phase='idle';e.aim=null;e.nextShot=s.time+3.2;engaged--;}}
  }
 }
 bulletTick(){const s=this.s;s.bullets=s.bullets.filter(b=>{const next=point(b.x+b.vx*DT,b.y+b.vy*DT,b.z+b.vz*DT);let wall=Infinity;for(const o of BLOCKS){const h=segmentBox(b,next,o);if(h!==null)wall=Math.min(wall,h);}const hit=segmentBox(b,next,{x:s.x,y:s.y+.9,z:s.z,w:.64,h:1.8,d:.64});if(hit!==null&&hit<wall){s.health=Math.max(0,s.health-8);s.hurtAt=s.time;s.events.push({kind:'player-hit'});return false;}if(wall!==Infinity)return false;Object.assign(b,next);return s.time-b.born<4;});}
 spawn(wave){for(const e of this.s.enemies.filter(e=>e.wave===wave)){e.active=true;e.spawnAt=this.s.time;e.nextShot=this.s.time+.7+(e.id%2)*.5;}this.s.events.push({kind:'reinforcements',wave});}
 step(input={}){
  const s=this.s,prev=s.input;s.frame++;s.time=s.frame/FPS;s.events=[];s.input={forward:clamp(input.forward||0,-1,1),side:clamp(input.side||0,-1,1),lookX:clamp(input.lookX||0,-.2,.2),lookY:clamp(input.lookY||0,-.2,.2),fire:!!input.fire,reload:!!input.reload,flash:!!input.flash,jump:!!input.jump};s.move=[0,0];s.look=[0,0];if(s.won||s.dead)return s;
  const u=s.input;s.yaw=wrap(s.yaw+u.lookX);s.pitch=clamp(s.pitch+u.lookY,-.9,.8);s.look=[u.lookX/.07,-u.lookY/.07];const len=Math.max(1,Math.hypot(u.forward,u.side)),f=u.forward/len,r=u.side/len;s.move=[r,-f];this.move((Math.sin(s.yaw)*f+Math.cos(s.yaw)*r)*3.8*DT,(-Math.cos(s.yaw)*f+Math.sin(s.yaw)*r)*3.8*DT);
  if(u.jump&&!prev.jump&&s.y===0)s.vy=4.5;if(s.y>0||s.vy>0){s.vy-=12*DT;s.y=Math.max(0,s.y+s.vy*DT);if(s.y===0)s.vy=0;}
  if(s.reloadUntil&&s.time>=s.reloadUntil){const amount=Math.min(30-s.ammo,s.reserve);s.ammo+=amount;s.reserve-=amount;s.reloadUntil=0;}
  if(u.reload&&!prev.reload)this.reload();if(u.flash&&!prev.flash)this.throwFlash();
  if(u.fire&&!s.reloadUntil&&s.time-s.lastShotAt>=1/6-1e-8){if(s.ammo)this.shoot();else this.reload();}
  this.flashTick();this.enemyTick();this.bulletTick();s.trails=s.trails.filter(t=>s.time-t.time<.15);
  if(s.health===0){s.dead=true;return s;}
  if(s.sector===0&&s.kills>=2&&s.z<-5.8){s.sector=1;this.spawn(1);}
  if(s.sector===1&&s.kills>=4&&s.x>=12){s.sector=2;this.spawn(2);}
  if(s.sector===2&&s.kills>=6&&!s.enemies[6].active){s.nextWaveAt??=s.time+1.2;if(s.time>=s.nextWaveAt){this.spawn(3);s.nextWaveAt=null;}}
  if(s.route<ROUTE.length&&Math.hypot(s.x-ROUTE[s.route][0],s.z-ROUTE[s.route][1])<.55)s.route++;
  if(s.kills===8&&Math.hypot(s.x-EXIT.x,s.z-EXIT.z)<EXIT.r){s.exitProgress+=DT;if(s.exitProgress>=1){s.won=true;s.events.push({kind:'extracted'});}}else s.exitProgress=0;
  return s;
 }
 demoInput(){const s=this.s,eye=point(s.x,s.y+1.65,s.z);if(s.dead||s.won)return{};const e=this.living().filter(e=>!blocked(eye,point(e.x,1.3,e.z))&&Math.hypot(e.x-s.x,e.z-s.z)<23).sort((a,b)=>Math.hypot(a.x-s.x,a.z-s.z)-Math.hypot(b.x-s.x,b.z-s.z))[0];
  if(e){const dx=e.x-s.x,dz=e.z-s.z,d=Math.hypot(dx,dz),delta=wrap(Math.atan2(dx,-dz)-s.yaw),lookX=clamp(delta,-.07,.07),lookY=clamp(Math.atan2(1.3-eye.y,d)-s.pitch,-.04,.04);if(s.sector===1&&!s.flashCount&&Math.abs(delta)<.07)return{lookX,lookY,flash:true};return{lookX,lookY,fire:Math.abs(delta)<.07,reload:s.ammo===0};}
  const dest=ROUTE[Math.min(s.route,ROUTE.length-1)],dx=dest[0]-s.x,dz=dest[1]-s.z,d=Math.max(.1,Math.hypot(dx,dz)),angle=Math.atan2(dx,-dz);return{forward:(dx*Math.sin(s.yaw)-dz*Math.cos(s.yaw))/d,side:(dx*Math.cos(s.yaw)+dz*Math.sin(s.yaw))/d,lookX:clamp(wrap(angle-s.yaw),-.07,.07),lookY:clamp(-.04-s.pitch,-.04,.04),reload:s.ammo<12&&!s.reloadUntil};
 }
}
export const DEFAULTS={seed:28173,first:'A rifle held by two hands in the foreground of a desert alley. Armed soldiers stand ahead among wooden crates. An arched gateway, shuttered buildings and barrels line the alley.',global:'First-person rifle combat through a sunlit sandy desert alley, stone arch, courtyard and right-hand passage. Keep the same rifle, hands, architecture and wooden crates. Helmeted soldiers in green fatigues aim, fire short bursts, react to hits and fall when defeated. Cover blocks gunfire. Rifle shots show brief muzzle flashes, recoil, ejected casings and impact sparks. Flashbangs burst briefly and stunned soldiers stop firing. Keep the geography and enemy identities consistent, no extra soldiers or on-screen text, no HUD or blood.'};
export function dynamicPrompt(states){if(!states.length)return 'The rifle points along the alley toward two waiting soldiers.';const s=states.at(-1),events=states.flatMap(s=>s.events||[]),has=kind=>events.some(e=>e.kind===kind),out=[];
 out.push(['The player is in the entrance alley.','The player moves through the arched courtyard.','The player is in the right-hand stronghold.'][s.sector]);
 if(states.some(s=>Math.hypot(...s.move)>.01))out.push('The first-person viewpoint advances or sidesteps along the passage.');
 if(has('shot'))out.push('The rifle fires with brief muzzle flashes, recoil and ejected brass.');if(has('hit'))out.push('Bullets strike a soldier with small sparks and recoil.');if(has('kill'))out.push('A defeated soldier falls and remains on the ground.');if(has('reload')||s.reloadUntil)out.push('The rifle lowers while the player replaces its magazine.');
 if(has('flash-throw'))out.push('A small flashbang grenade is thrown forward.');if(has('flash-burst'))out.push('The grenade bursts in a brief white flash.');if(s.enemies.some(e=>e.blindUntil>s.time&&e.hp>0))out.push('Blinded soldiers shield their faces and cannot fire.');
 if(s.enemies.some(e=>e.phase==='windup'))out.push('An alerted soldier visibly raises and steadies the rifle before firing.');if(has('enemy-shot'))out.push('A soldier fires a short burst with visible orange bullet streaks toward an earlier player position.');if(has('reinforcements'))out.push('Two soldiers appear in the next area.');if(s.kills===8)out.push('All eight soldiers are down. A green extraction marker lights up in the far courtyard.');if(s.won)out.push('The player reaches the extraction point. The mission is complete.');if(s.dead)out.push('The player is defeated and the view settles low.');return out.join(' ');
}
