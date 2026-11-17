import { ONNX_FLOAT, ONNX_FLOAT16, parseProtoFields } from '../nn/onnxDequantFold.ts';

export type OnnxIoValue = { name: string; isTensor?: boolean; type?: string; shape?: readonly unknown[] };
export type OnnxModelIo = {
  inputNames: readonly string[];
  outputNames: readonly string[];
  inputMetadata: readonly OnnxIoValue[];
  outputMetadata: readonly OnnxIoValue[];
};

/** Resolve only named symbolic dimensions; never change bytes, numeric dimensions or ORT objects. */
export function applyOnnxFreeDimensionOverrides(io: OnnxModelIo, overrides?: Readonly<Record<string, number>>): OnnxModelIo {
  if (!overrides) return io;
  for (const [name, value] of Object.entries(overrides)) {
    if (!name || !Number.isSafeInteger(value) || value <= 0) throw new Error(`Invalid ONNX free dimension ${name}: ${value}`);
  }
  const resolve = (values: readonly OnnxIoValue[]) =>
    values.map((value) => ({
      ...value,
      shape: value.shape?.map((dim) => (typeof dim === 'string' && Object.hasOwn(overrides, dim) ? overrides[dim] : dim)),
    }));
  // InferenceSession's IO properties are prototype getters, not own enumerable fields.
  return { inputNames: io.inputNames, outputNames: io.outputNames, inputMetadata: resolve(io.inputMetadata), outputMetadata: resolve(io.outputMetadata) };
}

type Field = ReturnType<typeof parseProtoFields>[number];
const decoder = new TextDecoder();

/** Read declarations only: raw tensor payloads remain views, never copied or decoded. */
export function readOnnxModelIo(bytes: Uint8Array): OnnxModelIo {
  const fields = (field: Field) => parseProtoFields(bytes, field.valueStart, field.valueEnd);
  const text = (field: Field) => decoder.decode(bytes.subarray(field.valueStart, field.valueEnd));
  const graph = parseProtoFields(bytes).find((field) => field.field === 7 && field.wire === 2);
  if (!graph) throw new Error('onnx: missing graph');
  const graphFields = fields(graph);
  const initializers = new Set(
    graphFields
      .filter((field) => field.field === 5 && field.wire === 2)
      .map((field) => fields(field).find((item) => item.field === 8 && item.wire === 2))
      .filter((field): field is Field => !!field)
      .map(text),
  );
  const value = (field: Field): OnnxIoValue => {
    const info = fields(field);
    const name = info.find((item) => item.field === 1 && item.wire === 2);
    const type = info.find((item) => item.field === 2 && item.wire === 2);
    const tensor = type && fields(type).find((item) => item.field === 1 && item.wire === 2);
    if (!tensor) return { name: name ? text(name) : '', isTensor: false };
    const tensorFields = fields(tensor);
    const elementType = tensorFields.find((item) => item.field === 1 && item.wire === 0)?.varint;
    const shape = tensorFields.find((item) => item.field === 2 && item.wire === 2);
    return {
      name: name ? text(name) : '',
      isTensor: true,
      type: elementType === ONNX_FLOAT ? 'float32' : elementType === ONNX_FLOAT16 ? 'float16' : `tensor(${elementType})`,
      shape: shape
        ? fields(shape)
            .filter((item) => item.field === 1 && item.wire === 2)
            .map((item) => {
              const dim = fields(item);
              const size = dim.find((part) => part.field === 1 && part.wire === 0);
              const parameter = dim.find((part) => part.field === 2 && part.wire === 2);
              return parameter ? text(parameter) : (size?.varint ?? '');
            })
        : undefined,
    };
  };
  // Older ONNX exports also list initializer tensors as graph inputs; ORT excludes them.
  const inputMetadata = graphFields
    .filter((field) => field.field === 11 && field.wire === 2)
    .map(value)
    .filter((input) => !initializers.has(input.name));
  const outputMetadata = graphFields.filter((field) => field.field === 12 && field.wire === 2).map(value);
  return { inputNames: inputMetadata.map((input) => input.name), outputNames: outputMetadata.map((output) => output.name), inputMetadata, outputMetadata };
}
