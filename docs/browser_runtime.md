# Browser Runtime Plan

The shared engine substrate is backend-neutral:

1. chess rules and move encoding live in `src/chess/`;
2. feature encoding and `Evaluator` live in `src/nn/`;
3. search consumes only `BoardState` plus an `Evaluator`.

Research lanes may add:

- ONNX Runtime Web WASM evaluator;
- WebGPU/FP16 evaluator;
- worker message protocol and model cache;
- progressive loading from micro to balanced model.

Those lanes must not change move encoding or feature fixtures without a new benchmark id.
