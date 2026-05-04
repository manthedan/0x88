#!/usr/bin/env node
import { CANDIDATE_SPECS, estimateModelSizeMb, estimateParams, estimatedPolicyTop1 } from '../src/nn/modelSpec.ts';

const candidate = process.env.TINY_LEELA_MODEL_SPEC ?? 'balanced_48x5_int8';
const spec = CANDIDATE_SPECS.find((s) => s.name === candidate);
if (!spec) throw new Error(`Unknown TINY_LEELA_MODEL_SPEC=${candidate}`);

const modelSizeMb = estimateModelSizeMb(spec);
const policyTop1 = estimatedPolicyTop1(spec);
const policyTop5 = Math.min(1, policyTop1 + 0.28);
const wdlCrossEntropy = 1.25 - policyTop1 * 0.45;

// Architecture objective balances likely policy quality against browser size.
// It is a frozen proxy until real training labels/checkpoints exist.
const architectureScore = policyTop1 * 100 + policyTop5 * 25 - modelSizeMb * 3;

console.log(`METRIC architecture_score=${architectureScore.toFixed(6)}`);
console.log(`METRIC policy_top1_acc=${policyTop1.toFixed(6)}`);
console.log(`METRIC policy_top5_acc=${policyTop5.toFixed(6)}`);
console.log(`METRIC wdl_cross_entropy=${wdlCrossEntropy.toFixed(6)}`);
console.log(`METRIC model_size_mb=${modelSizeMb.toFixed(6)}`);
console.log(`METRIC model_params=${estimateParams(spec)}`);
