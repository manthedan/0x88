# BT4 SOAP remaining-300M c19000 final

Status: current `best-bt4` public candidate for 0x88 Neural.

## Artifact

- ONNX: `/models/bt4_soap_rem_c19000_final.onnx`
- Metadata: `/models/bt4_soap_rem_c19000_final.meta.json`
- SHA-256 ONNX: `de2dd68518f6e2ed162da5ee27c67a5a4795567d6e39972e3705824572b7660e`

## Architecture

BT4 / SquareFormer with ThreatGraph TG1 square attack-summary side input.

Short label:

```text
BT4-L6-W128-AH8-FF256-Hist7-RelBank-TG1
```

## Training-source summary

```text
Training data/teachers: LC0 public training data with Stockfish labels.
Not used: ChessBench, TCEC game corpus, or generic human-game corpus.
```

Stockfish is a teacher/label source; this model artifact does not bundle Stockfish code.

## Release note

This model is provided as the first 0x88 Neural BT4/SquareFormer public candidate. It replaces the older `bt4_anneal_muon_best` Tiny Leela demo artifact for public release planning, but the site may keep older artifacts available for rollback while the new model is wired and smoke-tested.
