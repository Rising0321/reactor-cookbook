import {combinePrompt} from '../runtime/prompts.js';
export {combinePrompt};
export const FPS=24,DT=1/FPS;
export const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
export const MOVES={
 jab:{wind:3,active:2,recovery:5,range:1.3,damage:6,stun:7,push:.12},
 cross:{wind:4,active:2,recovery:6,range:1.4,damage:8,stun:8,push:.18},
 spin:{wind:6,active:3,recovery:10,range:1.7,damage:12,stun:10,push:.35},
 uppercut:{wind:5,active:3,recovery:15,range:1.35,damage:12,stun:12,push:.12,cost:15,launch:6.2},
 wave:{wind:10,active:1,recovery:16,range:0,damage:14,stun:10,push:.3,cost:30},
};
const fighter=id=>({id,x:id?1.6:-1.6,y:0,vy:0,facing:id?-1:1,hp:100,energy:40,guard:100,blocking:false,stun:0,attack:null,combo:0,chainUntil:0,dash:0,dashDir:0,dashReady:0,walk:0,hitAt:-10,koAt:null,nextThink:0,decision:0,intent:{}});
export class ArcadeDuelGame{
 constructor(snapshot){this.s=snapshot?structuredClone(snapshot):{kind:'arcadeduel',frame:0,time:0,health:100,enemyHealth:100,won:false,dead:false,reason:null,ending:null,fighters:[fighter(0),fighter(1)],waves:[],waveId:0,hits:[],events:[],combo:0,bestCombo:0,lastHitFrame:-100,movesUsed:{},input:{},move:[0,0],look:[0,0]};}
 snapshot(){return structuredClone(this.s);}
 start(f,type){const m=MOVES[type];if(f.attack||f.stun||f.dash||f.blocking||f.energy<(m.cost||0))return false;f.energy-=m.cost||0;f.attack={type,age:0,hit:false,dir:f.facing};this.s.movesUsed[type]=(this.s.movesUsed[type]||0)+1;return true;}
 hit(a,b,type,dir){
  const s=this.s,base=MOVES[type],m=a.id?{...base,damage:Math.ceil(base.damage*.65)}:base;if(b.hp<=0||b.dash>0)return false;
  const block=b.blocking&&b.facing===-dir&&b.y===0;
  let result='hit';
  if(block){b.guard=Math.max(0,b.guard-m.damage*2.5);result=b.guard===0?'break':'block';b.stun=result==='break'?25:3;b.blocking=result!=='break';}
  else{b.hp=Math.max(0,b.hp-m.damage);b.stun=m.stun;b.attack=null;b.hitAt=s.time;b.x=clamp(b.x+dir*m.push,-3.55,3.55);if(m.launch)b.vy=m.launch;a.energy=clamp(a.energy+9,0,100);if(a.id===0){s.combo=s.frame-s.lastHitFrame<35?s.combo+1:1;s.lastHitFrame=s.frame;s.bestCombo=Math.max(s.bestCombo,s.combo);}else s.combo=0;}
  if(!block||result==='break')b.attack=null;
  s.hits.push({x:b.x-dir*.22,y:b.y+1.35,time:s.time,actor:a.id,type,result});s.events.push({actor:a.id,type,result});
  if(b.hp===0){b.koAt=s.time;b.vy=3.4;b.blocking=false;s.reason='ko';}
  return true;
 }
 ai(){
  const s=this.s,f=s.fighters[1],p=s.fighters[0];
  if(f.stun||f.attack||f.dash)return {};
  // Decisions use a sampled position every 1.25 seconds, never the player's buttons.
  if(s.frame<f.nextThink)return {side:f.intent.side||0,guard:!!f.intent.guard};
  f.nextThink=s.frame+30;
  const gap=Math.abs(p.x-f.x),toward=Math.sign(p.x-f.x),phase=f.decision++%6;
  if(gap>1.15){f.intent={side:toward};if(gap>2.4&&phase===2&&f.energy>=30){f.intent={};f.nextThink=s.frame+66;return {wave:true};}return f.intent;}
  f.intent={};f.nextThink=s.frame+60;
  if(phase===1){f.intent={guard:true};f.nextThink=s.frame+16;return f.intent;}
  return phase===3&&f.energy>=15?{upper:true}:{fire:true};
 }
 step(input={}){
  const s=this.s;s.frame++;s.time=s.frame/FPS;s.events=[];s.input={side:clamp(input.side||0,-1,1),fire:!!input.fire,jump:!!input.jump,guard:!!input.guard,upper:!!input.upper,wave:!!input.wave,dash:!!input.dash};s.move=[s.input.side,0];s.look=[0,0];
  s.hits=s.hits.filter(h=>s.time-h.time<.3);
  if(s.won||s.dead)return s;
  if(s.ending){
   for(const f of s.fighters){f.attack=null;f.walk=0;f.blocking=false;f.dash=0;if(f.y>0||f.vy>0){f.vy-=17*DT;f.y=Math.max(0,f.y+f.vy*DT);}}
   if(s.frame-s.ending.frame>=24){s.won=s.ending.playerWon;s.dead=!s.won;}
   return s;
  }
  const commands=[s.input,this.ai()];
  for(const f of s.fighters){
   const u=commands[f.id],other=s.fighters[1-f.id];f.walk=0;f.energy=clamp(f.energy+DT*5,0,100);f.guard=clamp(f.guard+DT*(f.blocking?0:18),0,100);
   if(f.stun>0)f.stun--;if(!f.attack&&!f.dash)f.facing=Math.sign(other.x-f.x)||f.facing;
   if(f.y>0||f.vy>0){f.vy-=17*DT;f.y=Math.max(0,f.y+f.vy*DT);if(f.y===0)f.vy=0;}
   f.blocking=!!u.guard&&f.y===0&&!f.attack&&!f.stun&&!f.dash&&f.guard>0;
   if(!f.stun&&!f.attack&&!f.blocking&&!f.dash){
    if(u.jump&&f.y===0){f.vy=6.7;f.y=.001;}
    if(u.dash&&s.frame>=f.dashReady&&f.energy>=10){f.energy-=10;f.dash=5;f.dashDir=u.side||f.facing;f.dashReady=s.frame+24;}
    else if(u.wave)this.start(f,'wave');else if(u.upper)this.start(f,'uppercut');else if(u.fire){if(s.frame>f.chainUntil)f.combo=0;this.start(f,['jab','cross','spin'][f.combo]);f.combo=(f.combo+1)%3;f.chainUntil=s.frame+40;}
   }
   if(f.dash){f.x+=f.dashDir*.32;f.dash--;}
   else if(!f.stun&&!f.attack){f.walk=(u.side||0)*(f.blocking?.35:1);f.x+=f.walk*(f.id?1.35:2.6)*DT;}
   f.x=clamp(f.x,-3.55,3.55);
  }
  const [a,b]=s.fighters,gap=b.x-a.x;
  if(Math.abs(gap)<.82&&Math.abs(a.y-b.y)<1.5){const d=Math.sign(gap)||1,mid=clamp((a.x+b.x)/2,-3.14,3.14);a.x=mid-d*.41;b.x=mid+d*.41;}
  for(const f of s.fighters){
   if(!f.attack)continue;const at=f.attack,m=MOVES[at.type],wind=m.wind+(f.id?30:0),other=s.fighters[1-f.id];at.age++;
   if(at.age>=wind&&!at.hit){
    if(at.type==='wave'){s.waves.push({id:s.waveId++,owner:f.id,x:f.x+at.dir*.55,y:f.y+1.2,dir:at.dir});at.hit=true;}
    else if(at.age<wind+m.active){const ahead=(other.x-f.x)*at.dir;if(ahead>=0&&ahead<=m.range&&Math.abs(other.y-f.y)<(at.type==='uppercut'?1.8:1)){at.hit=this.hit(f,other,at.type,at.dir);}}
   }
   if(at.age>=wind+m.active+m.recovery)f.attack=null;
  }
  s.waves=s.waves.filter(w=>{const old=w.x;w.x+=w.dir*6.5*DT;const a=s.fighters[w.owner],b=s.fighters[1-w.owner];if(b.x>=Math.min(old,w.x)-.4&&b.x<=Math.max(old,w.x)+.4&&Math.abs(b.y+1.2-w.y)<.85){if(this.hit(a,b,'wave',w.dir))return false;}return Math.abs(w.x)<4.4;});
  s.health=a.hp;s.enemyHealth=b.hp;
  if(a.hp<=0||b.hp<=0){s.ending={frame:s.frame,playerWon:b.hp<=0&&a.hp>0};s.reason='ko';}
  else if(s.time>=60){s.won=a.hp>b.hp;s.dead=!s.won;s.reason=a.hp===b.hp?'draw':'time';}
  if(s.frame-s.lastHitFrame>35)s.combo=0;
  return s;
 }
 demoInput(){const s=this.s,a=s.fighters[0],b=s.fighters[1],gap=Math.abs(a.x-b.x);return{side:gap>1.05?Math.sign(b.x-a.x):0,fire:gap<1.6,wave:gap>2&&a.energy>=30&&s.frame%24===0,upper:gap<1.3&&a.energy>=15&&s.frame%90===0};}
}
export const DEFAULTS={seed:13847,first:'Two martial artists facing each other on a raised fighting platform. Both fighters stand with their fists raised. Pillars stand behind the platform.',global:'A side-view arcade fighting match between exactly two martial artists, one in blue on the left and one in orange on the right, on a raised teal arena platform with dark pillars behind them. Keep their identities and the arena consistent. Full bodies stay visible. Fast punches, spinning kicks, rising uppercuts, jumping and brief dash afterimages follow the fighters. Crescent energy projectiles fly horizontally and disappear on impact. Brief sparks mark contact. A blocked strike meets raised forearms; a missed strike does not hurt the other fighter. No additional fighters, weapons, text or HUD.'};
export function dynamicPrompt(states){
 if(!states.length)return 'Both fighters face each other in ready stances.';
 const s=states.at(-1),events=states.flatMap(s=>s.events||[]),text=[];
 for(const id of [0,1]){const name=id?'The orange fighter':'The blue fighter',f=s.fighters[id],moves=[...new Set(states.map(s=>s.fighters[id].attack?.type).filter(Boolean))];if(moves.length)text.push(name+' performs '+moves.map(m=>({jab:'a quick jab',cross:'a straight cross',spin:'a spinning kick',uppercut:'a rising uppercut',wave:'a crescent energy wave'})[m]).join(' then ')+'.');if(f.blocking)text.push(name+' guards with raised forearms.');if(states.some(s=>s.fighters[id].dash))text.push(name+' dashes with brief afterimages.');if(f.y>.1)text.push(name+' is airborne.');if(f.walk)text.push(name+' steps '+(f.walk>0?'right':'left')+'.');}
 for(const e of events){const name=e.actor?'orange':'blue';text.push(e.result==='block'?`The ${name} attack is blocked with a small spark.`:e.result==='break'?`The ${name} attack breaks the opponent guard.`:`The ${name} attack connects with a bright impact spark and recoil.`);}
 if(s.won||s.dead||s.ending)text.push(s.reason==='ko'?((s.won||s.ending?.playerWon)?'The orange fighter is knocked out and falls. The blue fighter wins.':'The blue fighter is knocked out and falls. The orange fighter wins.'):'The round ends and both fighters stop attacking.');
 return [...new Set(text)].join(' ')||'The two fighters face each other in ready stances.';
}
