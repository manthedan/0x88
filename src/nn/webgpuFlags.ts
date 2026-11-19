// GPUBufferUsage / GPUMapMode values from the WebGPU spec. TypeScript's DOM lib declares the
// WebGPU interfaces but not these two globals, and Node (tests) has neither.
export const GPU_BUFFER_USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, STORAGE: 0x80 } as const;
export const GPU_MAP_MODE_READ = 0x1;
