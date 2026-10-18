export const FPS=24,DT=1/FPS;
export const PAD_DEFS=[{id:0,x:0,z:7,r:3,permanent:true},{id:1,x:-1.6,z:2,r:1.7},{id:2,x:1.6,z:-3,r:1.7},{id:3,x:-1.3,z:-8,r:1.7},{id:4,x:0,z:-13,r:3,permanent:true}];
export const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
export class FreezeGame{
 constructor(snapshot){this.s=snapshot?structuredClone(snapshot):{frame:0,time:0,x:0,y:1.04,z:7,yaw:Math.atan2(-1.6,5),pitch:-.31,vy:0,grounded:true,dead:false,won:false,landings:0,lastPad:0,pads:PAD_DEFS.map(p=>({...p,charge:p.permanent?1:0,expires:0})),events:[],input:{},target:1,beam:false,move:[0,0],look:[0,0]};}
 snapshot(){return structuredClone(this.s);}
 solid(p){return p.permanent||(p.charge>=1&&p.expires>this.s.time);}
 aim(){const s=this.s,f=[Math.sin(s.yaw)*Math.cos(s.pitch),Math.sin(s.pitch),-Math.cos(s.yaw)*Math.cos(s.pitch)];let best=null,score=.978;
  for(const p of s.pads.filter(p=>!p.permanent)){const d=[p.x-s.x,1.05-(s.y+1.62),p.z-s.z],len=Math.hypot(...d);if(len>13||len<1.3)continue;const dot=d.reduce((n,v,i)=>n+v*f[i]/len,0);if(dot>score){best=p;score=dot;}}
  return best;
 }
 step(input={}){const s=this.s;s.frame++;s.time=s.frame/FPS;s.events=[];s.input={forward:clamp(input.forward||0,-1,1),side:clamp(input.side||0,-1,1),lookX:clamp(input.lookX||0,-.2,.2),lookY:clamp(input.lookY||0,-.2,.2),fire:!!input.fire,jump:!!input.jump};const u=s.input;
  s.move=[0,0];s.look=[0,0];s.beam=false;if(s.dead||s.won)return this.snapshot();
  s.yaw=wrap(s.yaw+u.lookX);s.pitch=clamp(s.pitch+u.lookY,-1.05,.70);s.look=[clamp(u.lookX/.07,-1,1),clamp(-u.lookY/.07,-1,1)];
  for(const p of s.pads)if(!p.permanent&&p.charge>=1&&p.expires<=s.time){p.charge=0;s.events.push({kind:'ice-melted',target:p.id});}
  const target=this.aim();s.target=target?.id??null;s.beam=u.fire;
  if(u.fire&&target){const before=target.charge;target.charge=Math.min(1,target.charge+DT/.65);target.expires=s.time+12;if(before<1&&target.charge>=1)s.events.push({kind:'ice-created',target:target.id});}
  let forward=u.forward,side=u.side;const len=Math.max(1,Math.hypot(forward,side));forward/=len;side/=len;s.move=[side,forward?-forward:0];
  if(u.jump&&s.grounded){s.vy=6.5;s.grounded=false;s.events.push({kind:'jump'});}
  s.x+=(Math.sin(s.yaw)*forward+Math.cos(s.yaw)*side)*5.5*DT;s.z+=(-Math.cos(s.yaw)*forward+Math.sin(s.yaw)*side)*5.5*DT;
  const support=s.pads.find(p=>this.solid(p)&&Math.hypot(s.x-p.x,s.z-p.z)<p.r-.12);
  if(s.grounded&&!support){s.grounded=false;s.vy=0;}
  if(!s.grounded){s.vy-=13*DT;s.y+=s.vy*DT;if(s.vy<=0&&s.y<=1.04&&support){s.y=1.04;s.vy=0;s.grounded=true;s.landings++;s.events.push({kind:'landed',target:support.id});}else if(s.y<.25){s.y=.25;s.dead=true;s.events.push({kind:'fell-in-lava'});}}
  if(s.grounded&&support)s.lastPad=support.id;
  if(s.grounded&&support?.id===4){s.won=true;s.events.push({kind:'crossing-complete'});}
  return this.snapshot();
 }
 demoInput(){const s=this.s;if(s.dead||s.won)return{};const current=s.lastPad,target=s.pads[Math.min(4,current+1)];
  const dx=target.x-s.x,dz=target.z-s.z,distance=Math.hypot(dx,dz),yaw=Math.atan2(dx,-dz),pitch=Math.atan2(1.05-(s.y+1.62),Math.max(distance,.1));const lookX=clamp(wrap(yaw-s.yaw),-.06,.06),lookY=clamp(pitch-s.pitch,-.045,.045),delta=wrap(yaw-s.yaw);
  if(s.grounded&&!this.solid(target))return{lookX,lookY,fire:true};
  const moving=distance>.28;return{lookX,lookY,forward:moving?Math.cos(delta):0,side:moving?Math.sin(delta):0,jump:s.grounded&&Math.abs(delta)<.18};
 }
}
export const DEFAULTS={first:'A gun overlooks a lava lake inside a cavern. Small rocky outcrops form a route toward a raised platform.',global:'First-person traversal of a volcanic cavern above a lava lake. A freeze gun remains visible in the foreground. Small stepping stones lead toward a raised exit platform. Maintain the same cavern layout and weapon identity throughout.',seed:12345};
export function dynamicPrompt(states){const final=states.at(-1),events=states.flatMap(s=>s.events||[]),parts=[];if(!final)return 'The player stands still on the starting platform, looking across the lava lake.';
 if(states.some(s=>s.beam))parts.push(states.some(s=>s.beam&&s.target!==null)?'The gun fires a freezing beam at a nearby stepping stone.':'The gun fires a freezing beam into the aimed direction.');
 if(events.some(e=>e.kind==='ice-created'))parts.push('Ice finishes forming on the targeted stepping stone.');
 if(states.some(s=>!s.grounded&&!s.dead))parts.push('The player jumps across a gap over the lava.');else if(states.some(s=>Math.hypot(...s.move)>.05))parts.push('The player moves across the platform.');else parts.push('The player stays in place.');
 const yaw=states.reduce((n,s)=>n+(s.input?.lookX||0),0),pitch=states.reduce((n,s)=>n+(s.input?.lookY||0),0);if(Math.abs(yaw)>.04)parts.push(`The view turns ${yaw>0?'right':'left'}.`);if(Math.abs(pitch)>.04)parts.push(`The view tilts ${pitch>0?'up':'down'}.`);
 if(events.some(e=>e.kind==='landed'))parts.push('The player lands on a solid platform.');if(events.some(e=>e.kind==='ice-melted'))parts.push('An ice stepping stone melts back into the lava.');if(final.dead)parts.push('The player falls into the lava and stops.');if(final.won)parts.push('The player reaches the exit platform and stops.');return parts.join(' ');
}
export {combinePrompt} from '../runtime/prompts.js';
