# webui

The chat UI served at `/` by `dsx-server` is [llama.cpp](https://github.com/ggml-org/llama.cpp)'s web UI
(`tools/ui`), by Georgi Gerganov and the ggml authors, MIT licensed: see [LICENSE-llama.cpp](LICENSE-llama.cpp).
`dist/` is its build output at the commit pinned in `build.sh`, embedded into the server binary at compile time.
`logo.svg` (the dsx turbine) replaces llama.cpp's logo before the build, which also generates the favicons and app
icons from it.

`monitor.html` is dsx's own page, the monitor served at `/monitor`: a single self-contained file with no build
step, embedded directly. `build.sh` rebuilds only `dist/`, so it leaves `monitor.html` untouched.

```bash
./build.sh            # rebuild dist/ from the pinned llama.cpp commit (needs git, node 20+, npm)
./build.sh <commit>   # or from another commit
```
