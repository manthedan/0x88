enable subgroups;

@group(0) @binding(0) var<storage, read_write> T_add : array<f32>;
@group(0) @binding(1) var<storage, read> lv362 : array<f32>;
@group(0) @binding(2) var<storage, read> param_0 : array<f32>;
@group(0) @binding(3) var<storage, read> param_1 : array<f32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> red_result : array<f32, 1>;
var<workgroup> red_buf_staging : array<f32, 2>;
@compute @workgroup_size(64, 1, 1)
fn fused_matmul12_add9_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var matmul_rf_local : array<f32, 1>;
  var param_0_local : array<f32, 2>;
  var matmul_rf_local_1 : array<f32, 1>;
  matmul_rf_local[0i] = 0.000000e+00f;
  let cse_v1 : i32 = (i32(threadIdx.x) * 2i);
  param_0_local[0i] = param_0[(i32(threadIdx.x) * 2i)];
  param_0_local[1i] = param_0[((i32(threadIdx.x) * 2i) + 1i)];
  let cse_v2 : i32 = ((v__1 * 128i) + (i32(threadIdx.x) * 2i));
  matmul_rf_local[0i] = fma(lv362[((v__1 * 128i) + (i32(threadIdx.x) * 2i))], param_0_local[0i], matmul_rf_local[0i]);
  matmul_rf_local[0i] = fma(lv362[(((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + 1i)], param_0_local[1i], matmul_rf_local[0i]);
  matmul_rf_local_1[0i] = 0.000000e+00f;
  matmul_rf_local_1[0i] = (matmul_rf_local_1[0i] + matmul_rf_local[0i]);
  var t0 : array<f32, 1>;
  var red_buf0 : array<f32, 1>;
  var t0_1 : array<f32, 1>;
  var red_buf0_1 : array<f32, 1>;
  red_buf0[0i] = matmul_rf_local_1[0i];
  t0[0i] = subgroupShuffleDown(red_buf0[0i], u32(16i));
  red_buf0[0i] = (red_buf0[0i] + t0[0i]);
  t0[0i] = subgroupShuffleDown(red_buf0[0i], u32(8i));
  red_buf0[0i] = (red_buf0[0i] + t0[0i]);
  t0[0i] = subgroupShuffleDown(red_buf0[0i], u32(4i));
  red_buf0[0i] = (red_buf0[0i] + t0[0i]);
  t0[0i] = subgroupShuffleDown(red_buf0[0i], u32(2i));
  red_buf0[0i] = (red_buf0[0i] + t0[0i]);
  t0[0i] = subgroupShuffleDown(red_buf0[0i], u32(1i));
  red_buf0[0i] = (red_buf0[0i] + t0[0i]);
  if ((i32(threadIdx.x) % 32i) == 0i) {
    red_buf_staging[(i32(threadIdx.x)>>5u)] = red_buf0[0i];
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i) {
    red_buf0_1[0i] = red_buf_staging[i32(threadIdx.x)];
  }
  t0_1[0i] = subgroupShuffleDown(red_buf0_1[0i], u32(1i));
  red_buf0_1[0i] = (red_buf0_1[0i] + t0_1[0i]);
  if (i32(threadIdx.x) == 0i) {
    red_result[0i] = red_buf0_1[0i];
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i) {
    T_add[v__1] = (red_result[0i] + param_1[0i]);
  }
}
