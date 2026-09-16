"""Stage packing; retain four independent denoising KV histories.

Transform the pinned upstream method in memory, without modifying its checkout.
Used by the GPU-count benchmarks and by the opt-in three-GPU turbo serving mode
(``liveavatar_turbo``); the released five-GPU backend leaves it untouched. A
physical DiT rank executes consecutive denoising stages, each with its own
unchanged-size cache. One GPU additionally executes streaming VAE when there is
no separate VAE rank.
"""

import inspect
import random
import textwrap


class StageRandomStreams:
    """Preserve native per-rank Python RNG used by conditional RoPE offsets."""

    def __init__(self, stages):
        self.states = dict.fromkeys(stages, random.getstate())

    def call(self, stage, forward, *args, **kwargs):
        outside = random.getstate()
        random.setstate(self.states[stage])
        try:
            return forward(*args, **kwargs)
        finally:
            self.states[stage] = random.getstate()
            random.setstate(outside)


def stage_groups(world_size):
    if world_size not in range(1, 6):
        raise ValueError("Expected one through five GPUs")
    ranks = max(1, world_size - 1)
    # Contiguous stages, balanced as evenly as possible.
    return [
        list(range((rank * 4) // ranks, ((rank + 1) * 4) // ranks))
        for rank in range(ranks)
    ]


def install_grouped_generate(model_class, world_size, shared_vae=False):
    if shared_vae and world_size > 4:
        raise ValueError("At most four DiT ranks can share the VAE")
    shared_vae = shared_vae or world_size == 1
    groups = stage_groups(world_size + 1 if shared_vae else world_size)
    original = model_class.generate
    source = textwrap.dedent(inspect.getsource(original))

    def replace(old, new, count=1):
        nonlocal source
        actual = source.count(old)
        if actual != count:
            raise RuntimeError(f"Upstream source mismatch ({actual} != {count}): {old}")
        source = source.replace(old, new)

    replace(
        "in_dit_device = dist.get_rank() < num_gpus_dit",
        "in_dit_device = dist.get_rank() < num_gpus_dit\n"
        f"    owned_steps = {groups!r}[dist.get_rank()] "
        "if in_dit_device else []\n"
        "    self._stage_kv = {}\n    self._stage_cross = {}\n"
        "    stage_random = StageRandomStreams(owned_steps)",
    )
    # The single-GPU case has one DiT rank and the same rank also decodes.
    replace("num_gpus_dit-1+int(enable_vae_parallel)", str(world_size - 1), count=6)
    if shared_vae:
        replace(
            "ref_latents = torch.empty_like(ref_latents).type_as(clip_latents[0])\n"
            f"                    dist.broadcast(ref_latents, src={world_size - 1})",
            f"if dist.get_rank() == {world_size - 1}:\n"
            "                        ref_latents = decoded_anchor.type_as(clip_latents[0])\n"
            "                    else:\n"
            "                        ref_latents = torch.empty_like(ref_latents).type_as(clip_latents[0])\n"
            f"                        dist.broadcast(ref_latents, src={world_size - 1})",
        )
        replace(
            "ref_latents = block_latents.unsqueeze(0)[:,:,0:1]",
            "decoded_anchor = block_latents.unsqueeze(0)[:,:,0:1]",
        )
        replace(
            f"dist.broadcast(ref_latents.contiguous(), src={world_size - 1})",
            f"dist.broadcast(decoded_anchor.contiguous(), src={world_size - 1})",
        )
        replace(
            "self._initialize_comm_group(num_gpus_dit=num_gpus_dit, enable_vae_parallel=enable_vae_parallel)",
            "self._initialize_comm_group(num_gpus_dit=num_gpus_dit, enable_vae_parallel=False)",
        )
    replace(
        "if (r==0 or r==1) and (dist.get_rank() != " + str(world_size - 1) + "):",
        "if (r==0 or r==1) and in_dit_device:",
    )
    prefill = """self.noise_model( #update clean kv cache
                    [block_latents], t=timestep*0, **block_arg_c, 
                    kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                    current_start=block_index * self.num_frames_per_block * frame_seq_length,
                    current_end=(block_index + 1) * self.num_frames_per_block * frame_seq_length)"""
    replacement = """if not self._stage_kv:
                    for stage in owned_steps:
                        self._stage_kv[stage] = deepcopy(self.kv_cache1)
                        self._stage_cross[stage] = deepcopy(self.crossattn_cache)
                    self.kv_cache1 = self._stage_kv[owned_steps[0]]
                    self.crossattn_cache = self._stage_cross[owned_steps[0]]
                for stage in owned_steps:
                    stage_random.call(stage, self.noise_model,
                        [block_latents], t=timestep*0, **block_arg_c,
                        kv_cache=self._stage_kv[stage], crossattn_cache=self._stage_cross[stage],
                        current_start=block_index * self.num_frames_per_block * frame_seq_length,
                        current_end=(block_index + 1) * self.num_frames_per_block * frame_seq_length)"""
    replace(prefill, replacement)
    replace("if i != dist.get_rank():", "if i not in owned_steps:")
    replace(
        "noise_pred_cond = self.noise_model(",
        "noise_pred_cond = stage_random.call(i, self.noise_model,",
    )
    replace(
        "if self.src_gpu is None:",
        "sample_scheduler._step_index = i\n"
        "                    if self.src_gpu is None or i != owned_steps[0]:",
    )
    replace(
        "kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,",
        "kv_cache=self._stage_kv[i], crossattn_cache=self._stage_cross[i],",
    )
    replace(
        "if self.tgt_gpu is None:",
        "if self.tgt_gpu is None or i != owned_steps[-1]:",
    )
    if shared_vae:
        replace(
            f"if enable_vae_parallel and dist.get_rank() == {world_size - 1}:\n                        vae_wait_start",
            "if False:  # Decoder shares this rank; no receive needed.\n                        vae_wait_start",
        )
    namespace = dict(original.__globals__, StageRandomStreams=StageRandomStreams)
    # Only the locally validated, pinned upstream function is compiled.
    exec(compile(source, "<liveavatar-grouped-benchmark>", "exec"), namespace)  # noqa: S102
    model_class.generate = namespace[original.__name__]
    return groups
