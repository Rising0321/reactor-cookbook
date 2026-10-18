export {GodzillaGame as Game,FPS,DEFAULTS,dynamicPrompt,combinePrompt} from './game.js';
export {createScene,drawHUD} from './scene.js';
export const config={
 id:'godzilla',healthMax:100,resource:{key:'heat',label:'HEAT',max:100,color:'#ffb55c'},
 storageKey:'godzilla-cover-v1',latest:'/assets/games/godzilla/latest.json',
 title:'Burn the skyline.',eyebrow:'INTERACTIVE RENDER LAB / 05',logo:'G↗',
 objective:'BREAK THE CITY DEFENSES',instruction:'Burn six bronze towers. Dodge their shells.',
 action:'Breath',jumpAction:'Stomp',lookAction:'Aim',crosshair:false,
 liveStatus:s=>`${s.targets} / 6 TOWERS · ${s.overheated?'COOLING':s.time-s.stompAt>=3.5?'STOMP READY':'STOMP '+(3.5-s.time+s.stompAt).toFixed(1)+'s'}`,
 statLabels:['TOWERS DESTROYED','HEALTH'],sample:'/assets/games/godzilla/sample.webm',
 validSnapshot:s=>s?.kind==='godzilla'&&Number.isFinite(s.x)&&Number.isFinite(s.heat)&&s.buildings?.length===19,
 stats:s=>[s.targets+' / 6',s.health+' / 100'],
 won:['CITY DEFENSES DESTROYED','The skyline is yours.','All six defense towers are rubble. Ready for another rampage?'],
 dead:['TRY AGAIN','The city fought back.','Dodge the shells. Stomp nearby towers while your breath cools.'],
 wonStatus:'All six defense towers destroyed.',deadStatus:'You were defeated.',
};
