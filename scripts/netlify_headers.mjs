// The header policy this site deploys is declared once, in netlify.toml, and
// public/_headers is generated from it. This module is the shared model both
// the generator and the policy checker read.
//
// TOML is parsed by smol-toml rather than by hand. An earlier version of this
// file implemented a "narrow TOML subset" instead, and five rounds of review
// found a new corner of the grammar wrong each time: comments after table
// headers (which silently dropped a whole block from the generated file),
// escaped quotes, multiline delimiters, table boundaries. None of those are
// interesting problems, and none of them are this project's problem to solve.
//
// Netlify semantics still modelled here (docs.netlify.com/manage/routing/headers):
//   - field names are case insensitive;
//   - a `_headers` field repeated under one path is concatenated per RFC 7230;
//   - multi-value headers may use multiline TOML strings.
import { parse } from 'smol-toml';

// Must be a genuine TOML table, not merely object-shaped. smol-toml returns
// `TomlDate` instances for date values, so `values = 2026-01-01` would otherwise
// pass as a table, yield no entries, and silently drop the block from the
// generated file with the staleness check blessing the omission.
const isPlainTable = (value) =>
  typeof value === 'object' && value !== null && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null);

// Netlify header values are strings. A TOML number, boolean, array, or table
// under [headers.values] is not a header this policy knows how to serve, and
// coercing one would publish something nobody wrote.
function readEntries(values, path, onInvalid) {
  // `values = true` would make Object.entries yield nothing, so the block would
  // vanish from the generated file with the staleness check blessing the
  // omission; `values = ["a"]` would invent a header named "0". Neither shape
  // is a header table, and neither may be read as one.
  if (values === undefined) return [];
  if (!isPlainTable(values)) {
    onInvalid(`${path}: [headers.values] must be a table, got ${Array.isArray(values) ? 'array' : typeof values}`);
    return [];
  }
  const entries = [];
  for (const [name, value] of Object.entries(values)) {
    if (typeof value !== 'string') {
      onInvalid(`${path}: ${name} must be a string, got ${Array.isArray(value) ? 'array' : typeof value}`);
      continue;
    }
    // A multiline value carries newlines that a header line cannot. Collapse
    // ONLY the line breaks and their indentation — internal runs of spaces are
    // part of the value (`filename="a  b.txt"`) and must survive.
    entries.push({ name, value: value.replace(/\s*\n\s*/g, ' ').trim() });
  }
  return entries;
}

// Throws on malformed TOML, which is the correct outcome: netlify.toml that
// Netlify itself could not read must not pass a gate that claims to verify it.
// The parsed shape is validated too — a real parser guarantees valid TOML, not
// that the document says what this policy expects it to say.
export function parseHeaderBlocks(tomlText, onInvalid = () => {}) {
  const config = parse(tomlText);
  const tables = config.headers;
  if (tables === undefined) return [];
  if (!Array.isArray(tables)) {
    onInvalid('`headers` must be an array of [[headers]] tables');
    return [];
  }
  return tables.flatMap((table, index) => {
    if (!isPlainTable(table)) {
      onInvalid(`headers[${index}] must be a table, got ${Array.isArray(table) ? 'array' : typeof table}`);
      return [];
    }
    if (table.for !== undefined && typeof table.for !== 'string') {
      onInvalid(`headers[${index}]: \`for\` must be a string, got ${typeof table.for}`);
      return [];
    }
    const path = table.for;
    return [{ path, entries: readEntries(table.values, path ?? `headers[${index}]`, onInvalid) }];
  });
}

export function valuesByName(entries) {
  const values = new Map();
  for (const { name, value } of entries) {
    const key = name.toLowerCase();
    if (!values.has(key)) values.set(key, []);
    values.get(key).push(value);
  }
  return values;
}

// Repeated fields are concatenated per RFC 7230.
export function effectiveValue(values, name) {
  return values.get(name)?.join(', ');
}

// RFC 9111: directive names are case insensitive and arguments may appear in
// quoted-string form, so `max-age="31536000"` is the same directive as
// `max-age=31536000`. Every occurrence is kept — collapsing duplicates to the
// last one let `max-age=31536000, immutable, max-age=0` read as mutable while
// the emitted header still carried the year-long policy.
export function cacheControlDirectives(value) {
  const directives = new Map();
  if (value === undefined) return directives;
  for (const part of value.split(',')) {
    const token = part.trim();
    if (!token) continue;
    const split = token.indexOf('=');
    const name = (split === -1 ? token : token.slice(0, split)).trim().toLowerCase();
    const argument =
      split === -1
        ? ''
        : token
            .slice(split + 1)
            .trim()
            .replace(/^"(.*)"$/, '$1');
    if (!directives.has(name)) directives.set(name, []);
    directives.get(name).push(argument);
  }
  return directives;
}

export function hasDirective(directives, name, argument) {
  return (directives.get(name) ?? []).includes(argument);
}

// `max-age=031536000` is the same year as `max-age=31536000`; comparing the
// argument as text let the leading zero carry a mutable path past the
// immutable prohibition. Parsed numerically instead. Returns the largest stated
// value so a prohibition can test a threshold rather than one exact spelling —
// `max-age=31536001` is not less dangerous than `max-age=31536000`.
export function maxSeconds(directives, name) {
  const parsed = (directives.get(name) ?? []).filter((argument) => /^\d+$/.test(argument)).map(Number);
  return parsed.length ? Math.max(...parsed) : undefined;
}

// A directive stated twice with different arguments has no single meaning, and
// caches may resolve it differently. It is rejected rather than interpreted.
export function conflictingDirectives(directives) {
  return [...directives].filter(([, args]) => new Set(args).size > 1).map(([name]) => name);
}

// The configured build command decides whether the deploy gate runs at all.
export function parseBuildTable(tomlText) {
  const build = parse(tomlText).build;
  return isPlainTable(build) ? build : undefined;
}

// Redirects are inspected as parsed tables, not by scanning text. TOML permits
// keys in any order, so a regex requiring `.br` to appear before `force = true`
// missed a redirect that simply declared them the other way round.
export function parseRedirects(tomlText) {
  const config = parse(tomlText);
  const redirects = config.redirects;
  return Array.isArray(redirects) ? redirects.filter(isPlainTable) : [];
}
