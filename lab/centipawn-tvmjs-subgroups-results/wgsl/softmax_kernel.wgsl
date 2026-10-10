enable subgroups;

@group(0) @binding(0) var<storage, read_write> T_softmax_norm : array<f32>;
@group(0) @binding(1) var<storage, read> lv98 : array<f32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

var<workgroup> red_result : array<f32, 1>;
var<workgroup> T_softmax_maxelem_shared : array<f32, 1>;
var<workgroup> red_result_1 : array<f32, 1>;
var<workgroup> T_softmax_expsum_shared : array<f32, 1>;
var<workgroup> red_buf_staging : array<f32, 2>;
var<workgroup> red_buf_staging_1 : array<f32, 2>;
@compute @workgroup_size(64, 1, 1)
fn softmax_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var in_thread_T_softmax_maxelem_shared : array<f32, 1>;
  var in_thread_T_softmax_expsum_shared : array<f32, 1>;
  in_thread_T_softmax_maxelem_shared[0i] = -3.402823e+38f;
  let cse_v1 : i32 = ((v__1 * 64i) + i32(threadIdx.x));
  in_thread_T_softmax_maxelem_shared[0i] = max(in_thread_T_softmax_maxelem_shared[0i], lv98[((v__1 * 64i) + i32(threadIdx.x))]);
  var t0 : array<f32, 1>;
  var red_buf0 : array<f32, 1>;
  var t0_1 : array<f32, 1>;
  var red_buf0_1 : array<f32, 1>;
  red_buf0[0i] = in_thread_T_softmax_maxelem_shared[0i];
  t0[0i] = subgroupShuffleDown(red_buf0[0i], u32(16i));
  red_buf0[0i] = max(red_buf0[0i], t0[0i]);
  t0[0i] = subgroupShuffleDown(red_buf0[0i], u32(8i));
  red_buf0[0i] = max(red_buf0[0i], t0[0i]);
  t0[0i] = subgroupShuffleDown(red_buf0[0i], u32(4i));
  red_buf0[0i] = max(red_buf0[0i], t0[0i]);
  t0[0i] = subgroupShuffleDown(red_buf0[0i], u32(2i));
  red_buf0[0i] = max(red_buf0[0i], t0[0i]);
  t0[0i] = subgroupShuffleDown(red_buf0[0i], u32(1i));
  red_buf0[0i] = max(red_buf0[0i], t0[0i]);
  if ((i32(threadIdx.x) % 32i) == 0i) {
    red_buf_staging[(i32(threadIdx.x)>>5u)] = red_buf0[0i];
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i) {
    red_buf0_1[0i] = red_buf_staging[i32(threadIdx.x)];
  }
  t0_1[0i] = subgroupShuffleDown(red_buf0_1[0i], u32(1i));
  red_buf0_1[0i] = max(red_buf0_1[0i], t0_1[0i]);
  if (i32(threadIdx.x) == 0i) {
    red_result[0i] = red_buf0_1[0i];
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i) {
    T_softmax_maxelem_shared[0i] = red_result[0i];
  }
  in_thread_T_softmax_expsum_shared[0i] = 0.000000e+00f;
  workgroupBarrier();
  in_thread_T_softmax_expsum_shared[0i] = (in_thread_T_softmax_expsum_shared[0i] + exp((lv98[((v__1 * 64i) + i32(threadIdx.x))] - T_softmax_maxelem_shared[0i])));
  var t0_2 : array<f32, 1>;
  var red_buf0_2 : array<f32, 1>;
  var t0_3 : array<f32, 1>;
  var red_buf0_3 : array<f32, 1>;
  red_buf0_2[0i] = in_thread_T_softmax_expsum_shared[0i];
  t0_2[0i] = subgroupShuffleDown(red_buf0_2[0i], u32(16i));
  red_buf0_2[0i] = (red_buf0_2[0i] + t0_2[0i]);
  t0_2[0i] = subgroupShuffleDown(red_buf0_2[0i], u32(8i));
  red_buf0_2[0i] = (red_buf0_2[0i] + t0_2[0i]);
  t0_2[0i] = subgroupShuffleDown(red_buf0_2[0i], u32(4i));
  red_buf0_2[0i] = (red_buf0_2[0i] + t0_2[0i]);
  t0_2[0i] = subgroupShuffleDown(red_buf0_2[0i], u32(2i));
  red_buf0_2[0i] = (red_buf0_2[0i] + t0_2[0i]);
  t0_2[0i] = subgroupShuffleDown(red_buf0_2[0i], u32(1i));
  red_buf0_2[0i] = (red_buf0_2[0i] + t0_2[0i]);
  if ((i32(threadIdx.x) % 32i) == 0i) {
    red_buf_staging_1[(i32(threadIdx.x)>>5u)] = red_buf0_2[0i];
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i) {
    red_buf0_3[0i] = red_buf_staging_1[i32(threadIdx.x)];
  }
  t0_3[0i] = subgroupShuffleDown(red_buf0_3[0i], u32(1i));
  red_buf0_3[0i] = (red_buf0_3[0i] + t0_3[0i]);
  if (i32(threadIdx.x) == 0i) {
    red_result_1[0i] = red_buf0_3[0i];
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i) {
    T_softmax_expsum_shared[0i] = red_result_1[0i];
  }
  workgroupBarrier();
  T_softmax_norm[((v__1 * 64i) + i32(threadIdx.x))] = (exp((lv98[((v__1 * 64i) + i32(threadIdx.x))] - T_softmax_maxelem_shared[0i])) / T_softmax_expsum_shared[0i]);
}
