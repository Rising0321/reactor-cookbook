import {combinePrompt} from '../runtime/prompts.js';
export {combinePrompt};
export const FPS=24, DT=1/FPS;
export const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
export const center=z=>Math.sin((8-z)*.11)*2.6+Math.sin((8-z)*.24)*.8;
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
export const SPIDERS=[3,-2.5,-8,-13.5,-19,-25].map((z,id)=>({id,x:center(z)+(id%2?.6:-.4),z,size:[.88,1,.95,1.1,.92,1.05][id]}));
export class SpiderGame{
 constructor(snapshot){this.s=snapshot?structuredClone(snapshot):{kind:'cave-spiders',frame:0,time:0,x:center(7),y:0,z:7,yaw:0,pitch:-.22,vy:0,grounded:true,health:100,kills:0,dead:false,won:false,attackAt:-10,impactDone:true,lastHit:null,hurtAt:-10,spiders:SPIDERS.map(e=>({...e,hp:2,active:false,deadAt:null,hitAt:-10,biteAt:null,nextBite:0,facing:Math.PI,walking:false})),events:[],input:{},move:[0,0],look:[0,0]};}
 snapshot(){return structuredClone(this.s);}
 target(){const s=this.s,cp=Math.cos(s.pitch),dir=[Math.sin(s.yaw)*cp,Math.sin(s.pitch),-Math.cos(s.yaw)*cp];return s.spiders.filter(e=>e.hp>0).map(e=>{const d=[e.x-s.x,.6-(s.y+1.65),e.z-s.z],distance=Math.hypot(...d);return{e,distance,dot:d.reduce((n,v,i)=>n+v*dir[i]/distance,0)};}).filter(t=>t.distance<3.3&&t.dot>.72).sort((a,b)=>a.distance-b.distance)[0]?.e||null;}
 step(input={}){
  const s=this.s;s.frame++;s.time=s.frame/FPS;s.events=[];s.input={forward:clamp(input.forward||0,-1,1),side:clamp(input.side||0,-1,1),lookX:clamp(input.lookX||0,-.2,.2),lookY:clamp(input.lookY||0,-.2,.2),fire:!!input.fire,jump:!!input.jump};s.move=[0,0];s.look=[0,0];if(s.dead||s.won)return this.snapshot();const u=s.input;
  s.yaw=wrap(s.yaw+u.lookX);s.pitch=clamp(s.pitch+u.lookY,-1.1,.7);s.look=[clamp(u.lookX/.07,-1,1),clamp(-u.lookY/.07,-1,1)];
  const len=Math.max(1,Math.hypot(u.forward,u.side)),f=u.forward/len,r=u.side/len;s.move=[r,-f];
  s.z=clamp(s.z+(-Math.cos(s.yaw)*f+Math.sin(s.yaw)*r)*3.2*DT,-29,11);
  s.x=clamp(s.x+(Math.sin(s.yaw)*f+Math.cos(s.yaw)*r)*3.2*DT,center(s.z)-3.5,center(s.z)+3.5);
  if(u.jump&&s.grounded){s.vy=4.5;s.grounded=false;s.events.push({kind:'jump'});}
  if(!s.grounded){s.vy-=12*DT;s.y=Math.max(0,s.y+s.vy*DT);if(s.y===0){s.vy=0;s.grounded=true;}}
  if(u.fire&&s.time-s.attackAt>=.8){s.attackAt=s.time;s.impactDone=false;s.events.push({kind:'axe-swing'});}
  // Impact happens once, during the downward stroke. Turning during windup changes the target.
  if(!s.impactDone&&s.time-s.attackAt>=.29){s.impactDone=true;const e=this.target();if(e){e.hp--;e.hitAt=s.time;e.biteAt=null;e.nextBite=s.time+.8;s.lastHit={id:e.id,time:s.time,x:e.x,y:.65,z:e.z};s.events.push({kind:'axe-hit',target:e.id,hp:e.hp});if(e.hp===0){e.deadAt=s.time;s.kills++;s.events.push({kind:'spider-killed',target:e.id});}}else s.events.push({kind:'axe-miss'});}
  for(const e of s.spiders){e.walking=false;if(e.hp<=0)continue;let dx=s.x-e.x,dz=s.z-e.z,d=Math.hypot(dx,dz);if(d<8)e.active=true;if(!e.active)continue;e.facing=Math.atan2(-dx,-dz);
   if(e.biteAt!==null){if(s.time-e.biteAt>=.55){if(d<2.05&&s.y<.65){s.health=Math.max(0,s.health-16);s.hurtAt=s.time;s.events.push({kind:'spider-bite',target:e.id,health:s.health});}else s.events.push({kind:'bite-dodged',target:e.id});e.biteAt=null;e.nextBite=s.time+1.35;}continue;}
   if(s.time-e.hitAt<.35)continue;
   if(d>1.8){const speed=Math.min(.95*DT,d-1.8),nx=e.x+dx/d*speed,nz=e.z+dz/d*speed;
    if(!s.spiders.some(o=>o.id!==e.id&&o.hp>0&&Math.hypot(nx-o.x,nz-o.z)<2.4)){e.x=nx;e.z=nz;e.walking=true;}
   }else if(s.time>=e.nextBite){e.biteAt=s.time;s.events.push({kind:'bite-windup',target:e.id});}
  }
  // Solid bodies prevent walking through a living spider.
  for(const e of s.spiders.filter(e=>e.hp>0)){const dx=s.x-e.x,dz=s.z-e.z,d=Math.hypot(dx,dz);if(d<1.35){s.x=e.x+(d?dx/d:0)*1.35;s.z=e.z+(d?dz/d:1)*1.35;}}
  s.x=clamp(s.x,center(s.z)-3.5,center(s.z)+3.5);s.z=clamp(s.z,-29,11);
  if(s.health===0){s.dead=true;s.events.push({kind:'player-defeated'});}else if(s.kills===s.spiders.length){s.won=true;s.events.push({kind:'cave-cleared'});}
  return this.snapshot();
 }
 demoInput(){const s=this.s;if(s.dead||s.won)return{};const e=s.spiders.filter(e=>e.hp>0).sort((a,b)=>Math.hypot(a.x-s.x,a.z-s.z)-Math.hypot(b.x-s.x,b.z-s.z))[0];const dx=e.x-s.x,dz=e.z-s.z,d=Math.hypot(dx,dz),delta=wrap(Math.atan2(dx,-dz)-s.yaw);return{lookX:clamp(delta,-.06,.06),lookY:clamp(Math.atan2(.6-s.y-1.65,d)-s.pitch,-.045,.045),forward:d>2.65?Math.cos(delta):0,side:d>2.65?Math.sin(delta):0,fire:d<3.05&&Math.abs(delta)<.35};}
}
export const DEFAULTS={first:'A large steel axe blade and its long wooden handle occupy the right foreground of a stone cave passage. The wooden handle extends below the bottom edge of the image. Only the axe is visible in the foreground, with no hands, gloves or arms. Large brown spiders stand on the rocky floor. Orange wall torches illuminate blue-grey rock walls and stalactites.',global:'First-person exploration and axe combat inside a torch-lit stone cave. A single steel axe with a long wooden handle stays in the lower right foreground between swings, its handle extending below the image. The axe moves without visible hands, gloves or arms. Large brown eight-legged spiders crawl on the floor. Maintain the same winding cave, warm orange torches, cool stone walls, weapon and creature identities. No spiderwebs, text or HUD in the scene.',seed:12345};
export function dynamicPrompt(states){const s=states.at(-1);if(!s)return 'The axe rests in the foreground while a spider waits ahead in the torch-lit passage.';const events=states.flatMap(s=>s.events||[]),has=k=>events.some(e=>e.kind===k),parts=[];
 if(states.some(s=>s.time-s.attackAt<.7))parts.push('The axe is raised, then swings downward across the view and returns to its resting position.');
 if(has('axe-hit'))parts.push('The blade strikes a nearby spider, which recoils.');if(has('axe-miss'))parts.push('The axe swings through empty space without hitting a spider.');if(has('spider-killed'))parts.push('The struck spider collapses onto the floor with its legs folded and remains motionless.');
 if(states.some(s=>s.spiders.some(e=>e.hp>0&&e.walking)))parts.push('A living spider crawls toward the player.');if(has('bite-windup'))parts.push('A nearby spider raises its front legs to prepare a bite.');if(has('spider-bite'))parts.push('The spider lunges and bites the player.');if(has('bite-dodged'))parts.push('The spider lunges but misses the player.');
 if(states.some(s=>Math.hypot(...s.move)>.05))parts.push('The player walks through the cave with the viewpoint following the movement.');else parts.push('The player remains in place.');if(states.some(s=>!s.grounded))parts.push('The viewpoint rises and falls as the player jumps.');
 const yaw=states.reduce((n,s)=>n+(s.input.lookX||0),0),pitch=states.reduce((n,s)=>n+(s.input.lookY||0),0);if(Math.abs(yaw)>.04)parts.push(`The view turns ${yaw>0?'right':'left'}.`);if(Math.abs(pitch)>.04)parts.push(`The view tilts ${pitch>0?'up':'down'}.`);if(s.won)parts.push('All six spiders are defeated. The player stops in the cleared cave.');if(s.dead)parts.push('The player is defeated and stops moving.');return parts.join(' ');
}
