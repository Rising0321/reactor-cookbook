export {ArcadeDuelGame as Game,FPS,DEFAULTS,dynamicPrompt,combinePrompt} from './game.js';
export {createScene,drawHUD} from './scene.js';
export const config={
 id:'arcadeduel',healthMax:100,healthLabel:'BLUE',resource:{key:'enemyHealth',label:'ORANGE',max:100,color:'#ffa456'},
 sideView:true,crosshair:false,storageKey:'arcadeduel-cover-v1',latest:'/assets/games/arcadeduel/latest.json',
 title:'Break the guard.',eyebrow:'INTERACTIVE RENDER LAB / 06',logo:'VS',
 objective:'BLUE VS ORANGE',instruction:'Hold F to combo · Hold G to guard.',action:'Combo',
 extraActions:[{action:'guard',codes:['KeyG'],key:'G',label:'Guard',held:true},{action:'upper',codes:['KeyE'],key:'E',label:'Uppercut · 15 energy'},{action:'wave',codes:['KeyR'],key:'R',label:'Wave · 30 energy'},{action:'dash',codes:['ShiftLeft','ShiftRight'],key:'SHIFT',label:'Dash · 10 energy'}],
 liveStatus:s=>{const f=s.fighters[0],e=s.fighters[1];return 'ENERGY '+Math.floor(f.energy)+' · GUARD '+Math.floor(f.guard)+' · '+Math.max(0,60-Math.floor(s.time))+'s'+(e.attack?' · CPU WINDUP / ATTACK':'')+(s.combo>1?' · '+s.combo+' HIT':'');},
 statLabels:['BEST COMBO','BLUE HEALTH'],sample:'/assets/games/arcadeduel/sample.webm',
 validSnapshot:s=>s?.kind==='arcadeduel'&&s.fighters?.length===2&&Number.isFinite(s.health),stats:s=>[s.bestCombo+' HITS',s.health+' / 100'],
 won:['ROUND WON','Blue wins.','Chain punches into spinning kicks. Try an uppercut or energy wave.'],
 dead:['ROUND OVER','Round over.','Guard the windup or dash away. Win by KO or lead in health at 60 seconds. Ties count as no win.'],
 wonStatus:'Blue wins the round.',deadStatus:'Round over. Play again.',
};
