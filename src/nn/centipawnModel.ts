import { clearLc0ModelCacheEntry, type Lc0ModelLoadResult, loadLc0ModelForOrt, modelFileName } from '../lc0/modelCache.ts';
import { isMetadataTrustedModel, isOnnxModelParseError } from '../lc0/onnxModelRecovery.ts';
import { createOrtSession } from './ortRuntime.ts';

const LOAD_OPTIONS = { cache: true, allowUnverified: false, requireManifestEntry: true, manifestUrl: '/models/centipawn.manifest.json' } as const;
const DEFAULT_DEPS = { load: loadLc0ModelForOrt, create: createOrtSession, clear: clearLc0ModelCacheEntry };

type OwnedModel = { input?: Lc0ModelLoadResult['model']; info: Omit<Lc0ModelLoadResult, 'model'> };
function ownModel({ model, ...info }: Lc0ModelLoadResult): OwnedModel {
  return { input: model, info };
}

/** Metadata/model IDs describe tensor semantics, not the identity of an explicit research ONNX override. */
export function isPinnedCentipawnOnnx(modelUrl: string): boolean {
  try {
    return modelFileName(modelUrl) === 's4fix_v_ema.onnx';
  } catch {
    return false;
  }
}

/** App-origin pins, verified-only Cache Storage, and one-shot trusted-cache parse repair. */
export async function createCentipawnOrtSession(modelUrl: string, deps = DEFAULT_DEPS, signal?: AbortSignal) {
  const loadOptions = signal ? { ...LOAD_OPTIONS, signal } : LOAD_OPTIONS;
  let owned = ownModel(await deps.load(modelUrl, loadOptions));
  const reloadModel = async () => {
    owned = ownModel(await deps.load(modelUrl, loadOptions));
    const input = owned.input!;
    owned.input = undefined;
    return input;
  };
  // Synchronous consumption keeps original bytes out of the suspended async
  // session-creation frame. Unfolded/provider fallback reacquires verified bytes.
  const create = () => {
    signal?.throwIfAborted();
    const input = owned.input!;
    owned.input = undefined;
    return deps.create(input, { signal, reloadModel, rejectModelParseErrors: () => isMetadataTrustedModel(owned.info) });
  };
  try {
    return await create();
  } catch (error) {
    if (!isOnnxModelParseError(error) || !isMetadataTrustedModel(owned.info)) throw error;
    await deps.clear(owned.info.url);
    owned = ownModel(await deps.load(modelUrl, { ...loadOptions, forceReload: true }));
    if (owned.info.verification !== 'hashed' || owned.info.sha256Valid !== true) throw new Error('Centipawn recovery requires freshly SHA-256-verified bytes');
    return create();
  }
}
