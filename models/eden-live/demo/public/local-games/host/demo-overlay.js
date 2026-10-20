// HUD is rendered separately from the proxy frames used to condition the model.
export function mount(root, {onRestart} = {}) {
  root.innerHTML = `<div class="objective"><strong></strong><small></small></div>
    <div class="crosshair" aria-hidden="true">+</div><div class="stats"></div>
    <div class="result" role="dialog" hidden><h2></h2><p></p><button type="button">Play again</button><small class="error" role="alert"></small></div>`;
  const objective = root.querySelector('.objective');
  const stats = root.querySelector('.stats');
  const result = root.querySelector('.result');
  const button = result.querySelector('button');
  let current;
  button.onclick = async () => {
    button.disabled = true;
    result.querySelector('.error').textContent = '';
    try {
      if (onRestart) onRestart();
      else {
        const response = await fetch('/ext/game2v/switch', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({slug: current.slug, restart: true}),
        });
        if (!response.ok) throw Error('Could not restart the game. Try again.');
      }
    } catch (error) { result.querySelector('.error').textContent = error.message; }
    finally { button.disabled = false; }
  };
  return {update(state) {
    if (!state) return;
    current = state;
    objective.querySelector('strong').textContent = state.objective;
    objective.querySelector('small').textContent = state.instruction;
    root.querySelector('.crosshair').hidden = !state.crosshair;
    const values = state.stats.map(item => `${item.label}: ${item.value}`);
    if (state.resource) values.push(`${state.resource.label}: ${state.resource.value} / ${state.resource.max}`);
    if (state.healthMax && !state.stats.some(item => item.label === 'HEALTH')) values.push(`HEALTH: ${state.health} / ${state.healthMax}`);
    if (state.status) values.push(state.status);
    stats.textContent = values.join('  ·  ');
    result.hidden = !state.outcome && !state.paused;
    result.querySelector('h2').textContent = state.outcome ? state.outcome[1] : 'Paused';
    result.querySelector('p').textContent = state.outcome ? state.outcome[2] : 'Press P or Escape to resume.';
    button.hidden = !state.outcome;
  }};
}
