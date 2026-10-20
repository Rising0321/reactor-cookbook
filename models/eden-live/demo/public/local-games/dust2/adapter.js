export {Dust2Game as Game,FPS,DEFAULTS,dynamicPrompt,combinePrompt} from './game.js';
export {createScene,drawHUD} from './scene.js';
export const config={
 id:'dust2',healthMax:100,resource:{key:'ammo',label:'AMMO',max:30,color:'#e9c285'},storageKey:'dust2-cover-v1',latest:'/assets/games/dust2/latest.json',
 title:'Clear the passage.',eyebrow:'INTERACTIVE RENDER LAB / 07',logo:'D2',lookAction:'Aim',objective:'CLEAR & EXTRACT',instruction:'Clear the alley, arch and right passage. Reach the green exit.',action:'Fire',
 extraActions:[{action:'reload',codes:['KeyR'],key:'R',label:'Reload'},{action:'flash',codes:['KeyE'],key:'E',label:'Flashbang'}],
 liveStatus:s=>{const hint=s.kills===8?'GREEN EXIT':s.kills===2&&s.sector===0?'THROUGH ARCH':s.kills===4&&s.sector===1?'RIGHT PASSAGE':['ALLEY','COURTYARD','STRONGHOLD'][s.sector];return hint+' · '+s.kills+'/8 · RESERVE '+s.reserve+' · FLASH '+s.flashes+(s.reloadUntil?' · RELOADING':s.enemies.some(e=>e.phase==='windup')?' · ENEMY AIMING':'');},
 statLabels:['ENEMIES CLEARED','HEALTH'],sample:'/assets/games/dust2/sample.webm',
 validSnapshot:s=>s?.kind==='dust2'&&s.enemies?.length===8&&Number.isFinite(s.ammo),stats:s=>[s.kills+' / 8',s.health+' / 100'],
 won:['MISSION COMPLETE','Extraction complete.','Eight enemies cleared. Try another route through the courtyards.'],dead:['TRY AGAIN','You were defeated.','Move during enemy windup, use crates for cover and flash before pushing.'],
 wonStatus:'All eight enemies cleared. Extraction complete.',deadStatus:'You were defeated.',
};
