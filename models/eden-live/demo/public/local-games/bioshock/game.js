import {combinePrompt} from '../runtime/prompts.js';
export {combinePrompt};
export const FPS=24, DT=1/FPS;
export const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
export const smooth=v=>{v=clamp(v);return v*v*(3-2*v);};
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
export const POOL={x:2.2,z:-10.5,rx:3,rz:2.4};
export const inPool=(x,z)=>((x-POOL.x)/POOL.rx)**2+((z-POOL.z)/POOL.rz)**2<=1;
export const ENEMIES=[{id:0,x:-1.3,z:3},{id:1,x:4.2,z:5.8},{id:2,x:1.4,z:-10.1,waterGuard:true},{id:3,x:3.3,z:-11.2,waterGuard:true}];
// The generator, boilers, walls and columns match the visible room geometry.
export const BLOCKS=[
 {x:1.3,y:1.3,z:-2.1,w:3.55,h:2.6,d:3.12},
 ...[[-5.8,5.8],[5.8,-11.8]].map(([x,z])=>({x,y:1.35,z,w:2.06,h:2.7,d:2.06})),
 ...[-7.7,7.7].map(x=>({x,y:2.9,z:-.5,w:.5,h:5.8,d:32})),
 ...[15.5,-16].map(z=>({x:0,y:2.9,z,w:16,h:5.8,d:.5})),
 ...[-7.2,7.2].flatMap(x=>[12,5,-2,-9,-15].map(z=>({x,y:2.8,z,w:.42,h:5.6,d:.48}))),
];
export function segmentBox(a,b,box,pad=0){
 let lo=0,hi=1;
 for(const [axis,size] of [['x','w'],['y','h'],['z','d']]){
  const min=box[axis]-box[size]/2-pad,max=box[axis]+box[size]/2+pad,d=b[axis]-a[axis];
  if(Math.abs(d)<1e-10){if(a[axis]<min||a[axis]>max)return null;continue;}
  let p=(min-a[axis])/d,q=(max-a[axis])/d;if(p>q)[p,q]=[q,p];lo=Math.max(lo,p);hi=Math.min(hi,q);if(lo>hi)return null;
 }return lo;
}
export const blocked=(a,b)=>BLOCKS.some(box=>segmentBox(a,b,box)!==null);
const point=(x,y,z)=>({x,y,z});
export class BioshockGame{
 constructor(snapshot){this.s=snapshot?structuredClone(snapshot):{
  kind:'bioshock',frame:0,time:0,x:-1.3,y:0,z:9,yaw:0,pitch:-.08,vy:0,grounded:true,
  health:100,eve:100,kills:0,dead:false,won:false,attackAt:-10,impactDone:true,
  shockAt:-10,cast:null,waterUntil:0,waterNextTick:0,lastHit:null,hurtAt:-10,
  swings:0,shocks:0,waterCasts:0,comboHits:0,
  enemies:ENEMIES.map(e=>({...e,hp:100,stunnedUntil:0,deathAt:null,hitAt:-10,lastKind:null,attackAt:null,nextAttack:0,facing:Math.PI,walking:false})),
  events:[],input:{},move:[0,0],look:[0,0],
 };}
 snapshot(){return structuredClone(this.s);}
 canStand(x,z,radius=.35,ignoreEnemy=null){
  return x>=-7.05&&x<=7.05&&z>=-15.3&&z<=14.8
   &&!BLOCKS.some(b=>Math.abs(x-b.x)<b.w/2+radius&&Math.abs(z-b.z)<b.d/2+radius)
   &&!this.s.enemies.some(e=>e.hp>0&&e.id!==ignoreEnemy&&Math.hypot(x-e.x,z-e.z)<radius+.4);
 }
 move(body,dx,dz,enemy=false){
  const n=Math.max(1,Math.ceil(Math.hypot(dx,dz)/.1));
  const allowed=(x,z)=>this.canStand(x,z,enemy ? .4 : .35,enemy?body.id:null)
   &&(!enemy||Math.hypot(x-this.s.x,z-this.s.z)>=1.15)
   &&(!body.waterGuard||inPool(x,z));
  for(let i=0;i<n;i++){if(allowed(body.x+dx/n,body.z))body.x+=dx/n;if(allowed(body.x,body.z+dz/n))body.z+=dz/n;}
 }
 aimEnemy(range,threshold){
  const s=this.s,from=point(s.x,s.y+1.68,s.z),cp=Math.cos(s.pitch),dir=[Math.sin(s.yaw)*cp,Math.sin(s.pitch),-Math.cos(s.yaw)*cp];
  return s.enemies.filter(e=>e.hp>0).map(e=>{const target=point(e.x,1.2,e.z),d=[target.x-from.x,target.y-from.y,target.z-from.z],distance=Math.hypot(...d);return{e,target,distance,dot:d.reduce((n,v,i)=>n+v*dir[i]/distance,0)};})
   .filter(t=>t.distance<=range&&t.dot>=threshold&&!blocked(from,t.target)).sort((a,b)=>a.distance-b.distance)[0]?.e||null;
 }
 shockTarget(){
  const s=this.s,from=point(s.x,s.y+1.68,s.z),cp=Math.cos(s.pitch),dir=point(Math.sin(s.yaw)*cp,Math.sin(s.pitch),-Math.cos(s.yaw)*cp);
  // Ground targeting uses an actual ray; looking at a guard targets the guard instead.
  const distance=dir.y<-.001?(.07-from.y)/dir.y:Infinity;
  const water=point(from.x+dir.x*distance,.07,from.z+dir.z*distance);
  const enemy=this.aimEnemy(10,.982),enemyDistance=enemy?Math.hypot(enemy.x-from.x,1.2-from.y,enemy.z-from.z):Infinity;
  if(distance<=10&&distance<enemyDistance&&inPool(water.x,water.z)&&!blocked(from,water))return{kind:'water',end:water};
  if(enemy)return{kind:'direct',enemy:enemy.id,end:point(enemy.x,1.2,enemy.z)};
  const end=point(from.x+dir.x*10,from.y+dir.y*10,from.z+dir.z*10);
  let limit=1;for(const b of BLOCKS){const hit=segmentBox(from,end,b);if(hit!==null)limit=Math.min(limit,hit);}
  if(dir.y<0)limit=Math.min(limit,Math.max(0,(.07-from.y)/(end.y-from.y)));
  return{kind:'miss',end:point(from.x+(end.x-from.x)*limit,from.y+(end.y-from.y)*limit,from.z+(end.z-from.z)*limit)};
 }
 hurtEnemy(e,amount,kind){
  if(e.hp<=0)return;const s=this.s;e.hp=Math.max(0,e.hp-amount);e.hitAt=s.time;e.lastKind=kind;
  s.lastHit={enemy:e.id,x:e.x,y:1.2,z:e.z,time:s.time,kind};s.events.push({kind:kind+'-hit',target:e.id,damage:amount,hp:e.hp});
  if(e.hp===0){e.deathAt=s.time;e.attackAt=null;s.kills++;s.events.push({kind:'enemy-killed',target:e.id});}
 }
 hurtPlayer(amount,kind){const s=this.s;s.health=Math.max(0,s.health-amount);s.hurtAt=s.time;s.events.push({kind,health:s.health});}
 step(input={}){
  const s=this.s,previous=s.input;s.frame++;s.time=s.frame/FPS;s.events=[];s.move=[0,0];s.look=[0,0];
  s.input={forward:clamp(input.forward||0,-1,1),side:clamp(input.side||0,-1,1),lookX:clamp(input.lookX||0,-.2,.2),lookY:clamp(input.lookY||0,-.2,.2),fire:!!input.fire,jump:!!input.jump,shock:!!input.shock};
  if(s.dead||s.won)return this.snapshot();const u=s.input;
  s.yaw=wrap(s.yaw+u.lookX);s.pitch=clamp(s.pitch+u.lookY,-1.1,.8);s.look=[clamp(u.lookX/.07,-1,1),clamp(-u.lookY/.07,-1,1)];
  const len=Math.max(1,Math.hypot(u.forward,u.side)),f=u.forward/len,r=u.side/len,oldX=s.x,oldZ=s.z;
  this.move(s,(Math.sin(s.yaw)*f+Math.cos(s.yaw)*r)*3.2*DT,(-Math.cos(s.yaw)*f+Math.sin(s.yaw)*r)*3.2*DT);
  if(Math.hypot(s.x-oldX,s.z-oldZ)>1e-8)s.move=[r,-f];
  if(u.jump&&!previous.jump&&s.grounded){s.vy=4.8;s.grounded=false;s.events.push({kind:'jump'});}
  if(!s.grounded){s.vy-=12*DT;s.y=Math.max(0,s.y+s.vy*DT);if(s.y===0){s.vy=0;s.grounded=true;}}
  if(s.cast&&s.time-s.shockAt>.45)s.cast=null;
  if(u.shock&&!previous.shock&&s.time-s.shockAt>=.9){
   if(s.eve>=20){s.eve-=20;s.shockAt=s.time;s.shocks++;s.cast=this.shockTarget();s.events.push({kind:'electro-bolt',target:s.cast.kind});
    if(s.cast.kind==='direct'){const e=s.enemies[s.cast.enemy];e.stunnedUntil=s.time+2.2;e.attackAt=null;this.hurtEnemy(e,10,'shock');}
    else if(s.cast.kind==='water'){if(s.waterUntil<=s.time)s.waterNextTick=s.time;s.waterUntil=s.time+1.6;s.waterCasts++;s.events.push({kind:'water-charged'});}
   }else s.events.push({kind:'eve-empty'});
  }
  if(s.time-s.shockAt>3)s.eve=Math.min(100,s.eve+6*DT);
  if(u.fire&&s.time-s.attackAt>=.85){s.attackAt=s.time;s.impactDone=false;s.swings++;s.events.push({kind:'wrench-swing'});}
  if(!s.impactDone&&s.time-s.attackAt>=.29){s.impactDone=true;const e=this.aimEnemy(2.8,.86);
   if(e){const combo=e.stunnedUntil>s.time;if(combo){s.comboHits++;s.events.push({kind:'combo-hit',target:e.id});}e.attackAt=null;e.nextAttack=s.time+.6;this.hurtEnemy(e,combo?60:35,'wrench');}
   else s.events.push({kind:'wrench-miss'});
  }
  if(s.waterUntil>s.time&&s.time>=s.waterNextTick){s.waterNextTick=s.time+.5;
   for(const e of s.enemies)if(e.hp>0&&inPool(e.x,e.z)){e.stunnedUntil=s.time+.65;e.attackAt=null;this.hurtEnemy(e,20,'water');}
   if(inPool(s.x,s.z)&&s.y<.3)this.hurtPlayer(12,'water-burn');
  }
  for(const e of s.enemies){e.walking=false;if(e.hp<=0||e.stunnedUntil>s.time)continue;
   const dx=s.x-e.x,dz=s.z-e.z,d=Math.hypot(dx,dz),clear=!blocked(point(e.x,1.2,e.z),point(s.x,s.y+1.2,s.z));
   if(e.attackAt!==null){if(s.time-e.attackAt>=.65){if(d<2.3&&s.y<.65&&clear)this.hurtPlayer(16,'enemy-strike');else s.events.push({kind:'strike-dodged',target:e.id});e.attackAt=null;e.nextAttack=s.time+1.25;}continue;}
   if(d>7.5||!clear)continue;e.facing=Math.atan2(-dx,-dz);
   if(d>1.8){const x=e.x,z=e.z;this.move(e,dx/d*1.15*DT,dz/d*1.15*DT,true);e.walking=Math.hypot(x-e.x,z-e.z)>1e-8;}
   else if(s.time>=e.nextAttack){e.attackAt=s.time;s.events.push({kind:'strike-windup',target:e.id});}
  }
  if(s.health===0){s.dead=true;s.events.push({kind:'player-defeated'});}
  else if(s.kills===s.enemies.length){s.won=true;s.events.push({kind:'room-cleared'});}
  return this.snapshot();
 }
 demoInput(){
  const s=this.s;if(s.dead||s.won)return{};
  const dry=s.enemies.filter(e=>e.hp>0&&!e.waterGuard).sort((a,b)=>Math.hypot(a.x-s.x,a.z-s.z)-Math.hypot(b.x-s.x,b.z-s.z))[0];
  const aim=(x,y,z)=>{const dx=x-s.x,dz=z-s.z,d=Math.hypot(dx,dz),delta=wrap(Math.atan2(dx,-dz)-s.yaw);return{lookX:clamp(delta,-.065,.065),lookY:clamp(Math.atan2(y-s.y-1.68,d)-s.pitch,-.045,.045),delta,d};};
  const moveTo=(x,z,look)=>{const dx=x-s.x,dz=z-s.z,n=Math.max(.1,Math.hypot(dx,dz));return{...look,forward:(dx*Math.sin(s.yaw)-dz*Math.cos(s.yaw))/n,side:(dx*Math.cos(s.yaw)+dz*Math.sin(s.yaw))/n};};
  if(dry){const a=aim(dry.x,1.2,dry.z);return{...a,forward:a.d>2.35?Math.cos(a.delta):0,side:a.d>2.35?Math.sin(a.delta):0,fire:a.d<2.6&&Math.abs(a.delta)<.12,shock:a.d<5.5&&Math.abs(a.delta)<.06&&dry.stunnedUntil<=s.time&&s.eve>=20&&s.time-s.shockAt>=.9};}
  // Walk around the generator on its left before aiming at the water surface.
  if(s.z>-4.7){const x=-2.8,z=s.x>-2.65?Math.max(s.z,1): -5.2;return moveTo(x,z,aim(x,1.3,z-2));}
  if(Math.hypot(s.x+.9,s.z+7.2)>.22)return moveTo(-.9,-7.2,aim(POOL.x-1.8,.07,POOL.z+1));
  const a=aim(POOL.x-1.8,.07,POOL.z+1);return{...a,shock:this.shockTarget().kind==='water'&&s.waterUntil<=s.time&&s.eve>=20&&!s.input.shock};
 }
}
export const DEFAULTS={
 first:'First-person view inside an underwater art deco machine room. A large orange steel pipe wrench held in the right hand fills the lower right foreground. An open left hand with faint blue electrical veins is visible at lower left. Two hostile masked workers in worn brown and purple work clothes stand ahead on teal floor tiles. Copper pipes, brass beams, boilers and warm lamps line the room. A blue underwater window is on the left wall and a large generator stands farther ahead on the right.',
 global:'First-person electro bolt and pipe wrench combat in an underwater art deco machine room. Keep the same orange pipe wrench in the right hand and bare electrical left hand visible, teal tiled walls and floor, copper pipes, brass trim, warm lamps, underwater window and central generator. Hostile masked workers approach and attack with their hands. Electric shocks are bright blue; stunned enemies tremble with blue arcs. A shallow pool at the back conducts electricity. Defeated enemies remain on the floor. No text or HUD in the scene.',seed:314159,
};
export function dynamicPrompt(states){
 const s=states.at(-1);if(!s)return 'The player holds the wrench ready in the underwater machine room.';
 const events=states.flatMap(s=>s.events||[]),has=k=>events.some(e=>e.kind===k),parts=[];
 if(has('electro-bolt')||states.some(s=>s.cast))parts.push('The left hand extends and releases a branching bright blue electrical bolt in the aimed direction.');
 if(has('shock-hit'))parts.push('The bolt strikes a masked enemy, interrupting its attack and stunning it with blue arcs.');
 if(has('wrench-swing')||states.some(s=>s.time-s.attackAt<.75))parts.push('The right hand raises the orange pipe wrench, swings it forward and returns it to the ready position.');
 if(has('wrench-hit'))parts.push('The wrench connects with a nearby enemy, which recoils from the impact.');
 if(has('wrench-miss'))parts.push('The wrench swings through empty space.');
 if(has('combo-hit'))parts.push('The stunned enemy takes a heavy wrench blow.');
 if(states.some(s=>s.waterUntil>s.time))parts.push('Bright blue electrical arcs spread across the shallow pool, shocking enemies standing in the water.');
 if(has('water-burn'))parts.push('The player steps into the electrified pool and is hurt by the current.');
 if(has('enemy-killed'))parts.push('A defeated masked enemy collapses onto the tiles and remains motionless.');
 if(states.some(s=>s.enemies.some(e=>e.walking)))parts.push('A masked enemy walks toward the player.');
 if(has('strike-windup'))parts.push('A nearby enemy raises its arms to prepare a close strike.');
 if(has('enemy-strike'))parts.push('The enemy strikes the player.');if(has('strike-dodged'))parts.push('The enemy strike misses as the player evades it.');
 if(states.some(s=>Math.hypot(...s.move)>.05))parts.push('The viewpoint moves through the machine room.');
 if(states.some(s=>!s.grounded))parts.push('The viewpoint rises and falls in a jump.');
 const yaw=states.reduce((n,s)=>n+(s.input.lookX||0),0),pitch=states.reduce((n,s)=>n+(s.input.lookY||0),0);
 if(Math.abs(yaw)>.04)parts.push(`The view turns ${yaw>0?'right':'left'}.`);if(Math.abs(pitch)>.04)parts.push(`The view tilts ${pitch>0?'up':'down'}.`);
 if(s.won)parts.push('All four enemies are defeated. The machine room is clear.');if(s.dead)parts.push('The player is defeated and stops moving.');
 return parts.join(' ')||'The player holds position with the pipe wrench and electrical hand ready.';
}
