<!--
  See CONTRIBUTING.md for the full workflow.
  Keep PRs small and focused — one issue per PR.
-->

## What changed

<!-- One or two sentences. What does this PR do? -->


## Why

<!-- Link the issue. Use "Closes #N" for a full fix, "Refs #N" for a checklist item. -->

Closes #


## How I verified it

<!--
  The actual commands you ran and what you saw. Not "tested locally".
  e.g. "Ran `python -m uvicorn main:app`, uploaded a chest X-ray, confirmed the
  analysis returned and no JS errors in console."
-->

```
```


## Checklist

- [ ] Branched off an up-to-date `main`
- [ ] `python -m py_compile main.py app.py src/*.py` passes
- [ ] App boots and the affected path works in the browser
- [ ] No secrets, weights, datasets, or runtime artifacts in the diff (`git diff --cached --stat`)
- [ ] Any number shown to the user comes from a measurement, not a hardcoded literal
- [ ] Any new user-facing claim is something the code actually verifies


## Notes for the reviewer

<!-- Anything non-obvious, tradeoffs you made, or parts you want a closer look at. -->
