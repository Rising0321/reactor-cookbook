# Models

Models you serve on Reactor. Every folder here is a `reactor` workspace: a
`reactor.yaml` naming the model and the GPU it wants, an adapter built on the
[Reactor Runtime](https://github.com/reactor-team/reactor-runtime), and an image
definition the CLI builds and runs. New recipes describe that image in
`reactor.yaml`'s `build:` block, which `reactor build` renders in memory; some
older recipes retain an explicit Dockerfile. `reactor build` and `reactor run`
serve either form on your own machine, and the
[build configuration](https://deploy-docs.reactor.inc/platform/build) explains
the YAML-native workflow.

An example folder is self-contained. Whatever it needs stays inside it —
configuration, pinned upstream revisions, and any client written to demonstrate
it — so the model and the code that exercises it are read, changed, and copied
together.

Name a folder for the model it serves, and give it a README covering:

- What it does and when you'd reach for it
- Prerequisites (versions, credentials, GPU, environment)
- How to run it
- Notes on anything surprising in the code

Nothing lives directly in this folder besides this file — every model gets its
own subfolder. Complete applications and demos using hosted models belong in
[`examples/`](../examples); robotics integrations belong in
[`robotics/`](../robotics).
