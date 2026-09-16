"""Shape-keyed CUDA replay for a causal VAE decoder with explicit cache state."""

import torch


def install_vae_graph(vae):
    original = vae.model.decoder.forward
    graphs = {}

    def forward(x, feat_cache=None, feat_idx=None):
        if feat_cache is None or not any(torch.is_tensor(t) for t in feat_cache):
            return original(x, feat_cache=feat_cache, feat_idx=feat_idx)

        def shape(t):
            return (tuple(t.shape), t.dtype) if torch.is_tensor(t) else t

        key = (
            shape(x),
            tuple(shape(t) for t in feat_cache),
            torch.is_autocast_enabled("cuda"),
        )
        if key not in graphs:
            if len(graphs) >= 4:
                return original(x, feat_cache=feat_cache, feat_idx=feat_idx)
            static_x = x.clone()
            inputs = [t.clone() if torch.is_tensor(t) else t for t in feat_cache]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    original(static_x, feat_cache=list(inputs), feat_idx=[0])
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            cache_out = list(inputs)
            index = [0]
            with torch.cuda.graph(graph):
                output = original(static_x, feat_cache=cache_out, feat_idx=index)
            graphs[key] = (graph, static_x, inputs, output, cache_out, index[0])
        graph, static_x, inputs, output, cache_out, index = graphs[key]
        static_x.copy_(x)
        for source, target in zip(feat_cache, inputs, strict=True):
            if torch.is_tensor(source):
                target.copy_(source)
        graph.replay()
        # Independent outputs prevent later replay from overwriting an earlier
        # decoded frame or the caller's causal state.
        feat_cache[:] = [t.clone() if torch.is_tensor(t) else t for t in cache_out]
        feat_idx[0] = index
        return output.clone()

    vae.model.decoder.forward = forward
    return graphs
