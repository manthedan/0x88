export interface DisposableResource {
  dispose(): void;
}

/**
 * Owns disposable resources keyed by a fully resolved runtime variant.
 *
 * Analysis and Arena both retain expensive WASM workers while their variants
 * remain selected. Keeping that lifecycle here prevents family-specific maps
 * from drifting on cache-key, reconciliation, or page teardown behavior.
 */
export class DisposableVariantPool<Variant, Resource extends DisposableResource> {
  private readonly resources = new Map<string, Resource>();
  private readonly cacheKey: (variant: Variant) => string;
  private readonly create: (variant: Variant) => Resource;

  constructor(cacheKey: (variant: Variant) => string, create: (variant: Variant) => Resource) {
    this.cacheKey = cacheKey;
    this.create = create;
  }

  getOrCreate(variant: Variant): Resource {
    const key = this.cacheKey(variant);
    const existing = this.resources.get(key);
    if (existing) return existing;
    const resource = this.create(variant);
    this.resources.set(key, resource);
    return resource;
  }

  peek(variant: Variant): Resource | undefined {
    return this.resources.get(this.cacheKey(variant));
  }

  values(): IterableIterator<Resource> {
    return this.resources.values();
  }

  /** Dispose resources whose resolved cache keys are no longer active. */
  retain(variants: Iterable<Variant>): void {
    const active = new Set(Array.from(variants, this.cacheKey));
    for (const [key, resource] of this.resources) {
      if (active.has(key)) continue;
      resource.dispose();
      this.resources.delete(key);
    }
  }

  disposeAll(): void {
    for (const resource of this.resources.values()) resource.dispose();
    this.resources.clear();
  }

  get size(): number {
    return this.resources.size;
  }
}
