import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test, { after } from 'node:test';
import { fileURLToPath } from 'node:url';

// GPU block constructors use parameter properties, requiring TS transforms.
if (!process.execArgv.includes('--experimental-transform-types')) {
  test('TVM kernel readbacks (re-run under --experimental-transform-types)', () => {
    const env = { ...process.env };
    delete env.NODE_TEST_CONTEXT;
    const child = spawnSync(process.execPath, ['--experimental-transform-types', '--no-warnings', '--test-reporter=tap', fileURLToPath(import.meta.url)], {
      encoding: 'utf8',
      env,
    });
    assert.equal(child.status, 0, `${child.stdout}\n${child.stderr}`);
    assert.match(child.stdout, /^# pass 24$/m);
    assert.match(child.stdout, /^# fail 0$/m);
  });
} else {
  const previousMapMode = Object.getOwnPropertyDescriptor(globalThis, 'GPUMapMode');
  Object.defineProperty(globalThis, 'GPUMapMode', { configurable: true, value: { READ: 1 } });
  after(() => {
    if (previousMapMode) Object.defineProperty(globalThis, 'GPUMapMode', previousMapMode);
    else delete globalThis.GPUMapMode;
  });
  const { SquareformerTvmFfnWebgpuBlock } = await import('../src/nn/squareformerTvmFfnWebgpu.ts');
  const { SquareformerTvmAttentionWebgpuBlock } = await import('../src/nn/squareformerTvmAttentionWebgpu.ts');
  const { SquareformerTvmLayerNormQkvWebgpuBlock } = await import('../src/nn/squareformerTvmLayerNormQkvWebgpu.ts');
  const { SquareformerTvmInputStemWebgpuBlock } = await import('../src/nn/squareformerTvmInputStemWebgpu.ts');
  const { SquareformerTvmLegalPolicyGatherWebgpuBlock } = await import('../src/nn/squareformerTvmLegalPolicyGatherWebgpu.ts');
  const { SquareformerTvmPolicyHeadWebgpuBlock } = await import('../src/nn/squareformerTvmPolicyHeadWebgpu.ts');
  const { SquareformerTvmValueHeadWebgpuBlock } = await import('../src/nn/squareformerTvmValueHeadWebgpu.ts');
  const { SquareformerTvmLayerNormFfnWebgpuBlock } = await import('../src/nn/squareformerTvmLayerNormFfnWebgpu.ts');
  const { SquareformerTvmAttnOutLnFfnWebgpuBlock } = await import('../src/nn/squareformerTvmAttnOutLnFfnWebgpu.ts');

  function readback({ failRead = false, map = async () => {} } = {}) {
    return {
      mapped: false,
      unmaps: 0,
      async mapAsync() {
        await map();
        this.mapped = true;
      },
      getMappedRange() {
        assert.equal(this.mapped, true);
        if (failRead) throw new Error('readback failed');
        return Float32Array.of(1, 2).buffer;
      },
      unmap() {
        this.unmaps += 1;
        this.mapped = false;
      },
    };
  }

  const blocks = [
    SquareformerTvmFfnWebgpuBlock,
    SquareformerTvmAttentionWebgpuBlock,
    SquareformerTvmLayerNormQkvWebgpuBlock,
    SquareformerTvmInputStemWebgpuBlock,
    SquareformerTvmLegalPolicyGatherWebgpuBlock,
    SquareformerTvmPolicyHeadWebgpuBlock,
  ];
  for (const Block of blocks) {
    for (const failRead of [false, true]) {
      test(`${Block.name} unmaps after ${failRead ? 'decode failure' : 'successful readback'}`, async () => {
        const buffer = readback({ failRead });
        const result = Block.prototype.readOutput.call({ readback: buffer, legalCount: 2 });
        if (failRead) await assert.rejects(result, /readback failed/);
        else assert.deepEqual(await result, Float32Array.of(1, 2));
        assert.equal(buffer.mapped, false);
        assert.equal(buffer.unmaps, 1);
      });
    }
    test(`${Block.name} does not read a failed mapping`, async () => {
      const buffer = readback({
        map: async () => {
          throw new Error('map failed');
        },
      });
      await assert.rejects(Block.prototype.readOutput.call({ readback: buffer, legalCount: 2 }), /map failed/);
      assert.equal(buffer.unmaps, 0);
    });
  }

  for (const failRead of [false, true]) {
    test(`value head unmaps both outputs after ${failRead ? 'decode failure' : 'success'}`, async () => {
      const wdlReadback = readback({ failRead });
      const qReadback = readback();
      const result = SquareformerTvmValueHeadWebgpuBlock.prototype.readOutput.call({ wdlReadback, qReadback });
      if (failRead) await assert.rejects(result, /readback failed/);
      else assert.deepEqual(await result, { wdl: Float32Array.of(1, 2), q: Float32Array.of(1, 2) });
      assert.equal(wdlReadback.unmaps, 1);
      assert.equal(qReadback.unmaps, 1);
    });
  }
  for (const failedOutput of ['wdlReadback', 'qReadback']) {
    test(`value head cleans a late map after ${failedOutput} mapping rejects`, async () => {
      let release;
      const gate = new Promise((resolve) => {
        release = resolve;
      });
      const late = readback({ map: () => gate });
      const failed = readback({
        map: async () => {
          throw new Error('map failed');
        },
      });
      const buffers = { wdlReadback: late, qReadback: late, [failedOutput]: failed };
      await assert.rejects(SquareformerTvmValueHeadWebgpuBlock.prototype.readOutput.call(buffers), /map failed/);
      release();
      await new Promise((resolve) => setImmediate(resolve));
      assert.equal(late.unmaps, 1);
      assert.equal(late.mapped, false);
    });
  }
  for (const Block of [SquareformerTvmLayerNormFfnWebgpuBlock, SquareformerTvmAttnOutLnFfnWebgpuBlock]) {
    test(`${Block.name} delegates readback cleanup to the FFN block`, async () => {
      const buffer = readback({ failRead: true });
      const ffn = { readOutput: () => SquareformerTvmFfnWebgpuBlock.prototype.readOutput.call({ readback: buffer }) };
      await assert.rejects(Block.prototype.readOutput.call({ ffn, lnFfn: ffn }), /readback failed/);
      assert.equal(buffer.unmaps, 1);
    });
  }
}
