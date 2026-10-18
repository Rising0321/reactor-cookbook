export {BioshockGame as Game,FPS,DEFAULTS,dynamicPrompt,combinePrompt} from './game.js';
export {createScene,drawHUD} from './scene.js';
export const config={
 id:'bioshock',healthMax:100,resource:{key:'eve',label:'EVE',max:100},
 storageKey:'bioshock-cover-v1',latest:'/assets/games/bioshock/latest.json',
 title:'Shock the depths.',eyebrow:'INTERACTIVE RENDER LAB / 04',logo:'B↯',
 objective:'CLEAR THE MACHINE ROOM',instruction:'Shock, then strike. Electrify water from dry ground.',
 action:'Wrench',extraActions:[{action:'shock',codes:['KeyE'],key:'E',label:'Electro bolt'}],
 statLabels:['ENEMIES DEFEATED','HEALTH'],sample:'/assets/games/bioshock/sample.webm',
 validSnapshot:s=>s?.kind==='bioshock'&&Number.isFinite(s.x)&&Number.isFinite(s.eve)&&s.enemies?.length===4,
 stats:s=>[s.kills+' / 4',s.health+' / 100'],
 won:['MACHINE ROOM CLEAR','All four enemies defeated.','Shock, strike and stay clear of the water. Try another route.'],
 dead:['TRY AGAIN','The depths got you.','Stun enemies before striking. Stay out of electrified water.'],
 wonStatus:'Machine room clear. All four enemies defeated.',deadStatus:'You were defeated.',
};
