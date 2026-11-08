import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { parse } from 'svelte/compiler';

const source = readFileSync(new URL('../src/routes/docs/+page.svelte', import.meta.url), 'utf8');
const ast = parse(source);
const elements = [];
function walk(node, ancestors = []) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'Element') elements.push({ node, ancestors });
  for (const [key, value] of Object.entries(node)) {
    if (key === 'parent') continue;
    if (Array.isArray(value)) value.forEach((child) => walk(child, [...ancestors, node]));
    else if (value && typeof value === 'object') walk(value, [...ancestors, node]);
  }
}
walk(ast.html);
const attr = (node, name) => node.attributes?.find((a) => a.type === 'Attribute' && a.name === name);
const textAttr = (node, name) => attr(node, name)?.value?.map((v) => v.data ?? '').join('');
const hasClass = (node, name) => textAttr(node, 'class')?.split(/\s+/).includes(name);
const byId = (id) => elements.find(({ node }) => textAttr(node, 'id') === id)?.node;
const descendants = (parent, name) => elements.filter(({ node, ancestors }) => ancestors.includes(parent) && (!name || node.name === name)).map(({ node }) => node);
const markup = (node) => source.slice(node.start, node.end);
function literal(node) {
  if (node.type === 'Literal') return node.value;
  if (node.type === 'ArrayExpression') return node.elements.map(literal);
  if (node.type === 'ObjectExpression') return Object.fromEntries(node.properties.map((p) => [p.key.name, literal(p.value)]));
  throw new Error(`Unexpected outline node: ${node.type}`);
}
const tocDeclaration = ast.instance.content.body.flatMap((n) => n.declarations ?? []).find((n) => n.id.name === 'toc');
const toc = literal(tocDeclaration.init);

test('the outline and article agree on seven peer sections and every topic is navigable in page order', () => {
  const sectionIds = elements.filter(({ node }) => node.name === 'section').map(({ node }) => textAttr(node, 'id'));
  assert.deepEqual(sectionIds, ['pages', 'analysis-in-depth', 'review-and-train', 'troubleshooting', 'engines', 'under-the-hood', 'licenses']);
  assert.deepEqual(toc.map((group) => group.id), sectionIds);
  const ids = elements.map(({ node }) => textAttr(node, 'id')).filter(Boolean);
  assert.equal(new Set(ids).size, ids.length, 'anchor IDs must be unique');
  for (const group of toc) {
    const section = byId(group.id);
    assert.equal(section.children.find((node) => node.name === 'h2')?.name, 'h2');
    const headingIds = descendants(section).filter((node) => /^h[34]$/.test(node.name)).map((node) => textAttr(node, 'id'));
    assert.deepEqual(group.items.flatMap((item) => [item.id, ...(item.children ?? []).map((child) => child.id)]), headingIds);
  }
});

test('every group has an explicit controlled disclosure, including all ten engine topics', () => {
  assert.equal(toc.find((group) => group.id === 'engines').items.length, 10);
  const toggle = elements.find(({ node }) => hasClass(node, 'group-toggle')).node;
  assert.ok(attr(toggle, 'aria-expanded'));
  assert.ok(attr(toggle, 'aria-controls'));
  assert.match(markup(toggle), /Show/);
  assert.match(source, /<ul id=\{`toc-\$\{group.id\}`\} hidden=\{!expandedGroups\[group.id\]\}/);
  assert.match(source, /aria-current=\{item.id === activeId \? 'location'/);
  assert.match(source, /group.id === activeGroupId \? 'true'/);
  assert.doesNotMatch(source, /group\.collapsed/);
  assert.doesNotMatch(markup(toggle), /group\.items\.length/, 'ambiguous topic counts are not shown');
  assert.match(source, /\.toc ul ul\{margin:0 0 0 12px\}/);
  assert.match(source, /\.toc-groups \.toc-group-heading \.toc-group-label\{[^}]*background:transparent/);
  assert.match(source, /\.group-toggle:hover\{background:transparent/);
  assert.doesNotMatch(source, /border-left|inset[^;]*[34]px 0/);
});

test('scroll tracking does not depend on animation frames being delivered', () => {
  assert.match(source, /addEventListener\('scroll', updateActive, \{ passive: true \}\)/);
  assert.match(source, /removeEventListener\('scroll', updateActive\)/);
  assert.doesNotMatch(source, /requestAnimationFrame/);
});

test('troubleshooting opens by default and common problems are linked outside Contents', () => {
  assert.match(source, /expandedGroups: Record<string, boolean> = \{ pages: true, troubleshooting: true \}/);
  const intro = elements.find(({ node }) => hasClass(node, 'doc-intro')).node;
  assert.deepEqual(descendants(intro, 'a').map((node) => textAttr(node, 'href')), ['#trouble-webgpu', '#trouble-downloads', '#trouble-cache']);
  for (const link of descendants(intro, 'a')) assert.ok(byId(textAttr(link, 'href').slice(1)));
});

test('mobile search is outside the collapsed panel and shares the desktop search focus action', () => {
  const trigger = elements.find(({ node }) => hasClass(node, 'mobile-search'));
  assert.ok(trigger);
  assert.ok(!trigger.ancestors.some((node) => textAttr(node, 'id') === 'toc-panel'));
  assert.match(markup(trigger.node), /Search docs/);
  assert.match(markup(trigger.node), /on:click=\{focusSearch\}/);
  assert.match(source, /if \(narrow\) tocOpen = true;\s+await tick\(\);\s+searchInput\?\.focus\(\)/);
  assert.match(source, /docsSearchShortcut\(navigator.platform\)/);
  assert.match(source, /event.key === '\/' && !typing/);
  assert.match(source, /event.metaKey \|\| event.ctrlKey/);
  assert.match(source, /let searchShortcut = 'Ctrl K'/);
});

test('all workspace launch links are with the title, ahead of the four-step procedure', () => {
  for (const workspace of ['analysis', 'arena', 'play']) {
    const heading = byId(`pages-${workspace}`);
    const card = elements.find(({ node, ancestors }) => node === heading && ancestors.some((parent) => hasClass(parent, 'page-block'))).ancestors.find((node) => hasClass(node, 'page-block'));
    const launch = descendants(card, 'a').find((node) => textAttr(node, 'href') === `/app/${workspace}/`);
    const steps = descendants(card, 'ol')[0];
    assert.ok(launch.end < steps.start);
    assert.doesNotMatch(markup(launch), /→/);
    assert.equal(steps.children.filter((node) => node.name === 'li').length, 4);
    assert.ok(elements.find(({ node, ancestors }) => node === launch && ancestors.some((parent) => hasClass(parent, 'task-heading'))));
    assert.ok(attr(heading, 'aria-labelledby'), 'the heading name excludes its permalink');
  }
  const analysisCard = elements.find(({ node }) => hasClass(node, 'page-block')).node;
  assert.equal(descendants(analysisCard, 'h4').length, 0, 'Analysis quick-start contains only its procedure');
  for (const id of ['exploring-results', 'explore-engine-lines', 'save-engine-setups', 'share-restore', 'explore-own-games', 'manage-collections']) {
    assert.ok(descendants(byId('analysis-in-depth')).includes(byId(id)), id);
  }
  for (const id of ['review-game', 'retry-mistakes', 'compare-critical-moments', 'export-review']) {
    assert.ok(descendants(byId('review-and-train')).includes(byId(id)), id);
  }
  for (const id of ['explore-own-games', 'compare-critical-moments']) {
    const topic = elements.find(({ node }) => node.children?.includes(byId(id))).node;
    assert.deepEqual(topic.children.filter((node) => node.type === 'Element').map((node) => node.name), ['h3', 'p', 'ol', 'p', 'p'], 'purpose, steps and two notes');
    assert.equal(descendants(topic, 'li').length, 4);
  }
  const links = elements.filter(({ node }) => node.name === 'a').map(({ node }) => textAttr(node, 'href')).filter((href) => /^https?:/.test(href));
  assert.equal(links.length, 76, 'all external link instances are retained');
  assert.equal(new Set(links).size, 41, 'all unique external destinations are retained');
  assert.ok(!elements.some(({ node }) => hasClass(node, 'not-yet')), 'obsolete availability note is removed');
});

test('beginner terms are defined inline without extending the quick-start list', () => {
  const section = markup(byId('pages'));
  assert.match(section, /Small net: a neural model/);
  assert.match(section, /game record \(PGN\) or position \(FEN\)/);
  assert.match(section, /its search limit, which is visits \(search iterations\) or depth depending on the engine/);
  const following = elements.find(({ node }) => node.children?.includes(byId('following-match'))).node;
  assert.equal(descendants(following, 'ul').length, 3, 'Resume and Saved results each have short sub-bullets');
  const note = descendants(following, 'p').find((node) => /Note:/.test(markup(node)));
  assert.match(markup(note), /parallel CPU runs/);
  assert.match(markup(note), /background completion is not guaranteed/);
  assert.match(markup(note), /App or engine-release changes/);
});

test('title hierarchy, quiet launch links and readable mobile contents are explicit', () => {
  assert.match(source, /\.doc-intro h1\{font-size:44px/);
  assert.match(source, /\.doc-content h2\{\s*font-size:28px/);
  assert.match(source, /\.doc-intro h1\{font-size:36px/);
  assert.match(source, /\.doc-content h2\{font-size:24px/);
  assert.match(source, /\.doc-content \.open-link\{min-height:44px/);
  assert.match(source, /\.toc-toggle-label\{[^}]*font-size:13px;[^}]*color:var\(--text\)/);
  assert.match(source, /background:transparent; color:var\(--ink-soft\); font-weight:600/);
});

test('subsection search containers exclude adjacent topics', () => {
  const containerNames = source.match(/const CONTAINERS = '([^']+)'/)[1].split(', ').map((name) => name.slice(1));
  for (const [id, className, excluded, included] of [
    ['cpu-wasm-relaxed-simd', 'runtime-subsection', /wasm32-wasip1/, /i32x4_relaxed_dot_i8x16_i7x16_add/],
    ['licenses-per-engine', 'source-links', /rights holders/, /corresponding-source\.tar\.gz/],
    ['explore-own-games', 'analysis-topic', /16 MiB|JSON backup/, /200 games/],
    ['compare-critical-moments', 'analysis-topic', /2 percentage points/, /400 visits/],
    ['maia-source-offer', 'callout', /Chessformer/, /derivation recipe/],
  ]) {
    const heading = byId(id);
    const ancestors = elements.find(({ node }) => node === heading).ancestors;
    const container = [...ancestors].reverse().find((node) => containerNames.some((name) => hasClass(node, name)));
    assert.ok(hasClass(container, className));
    assert.doesNotMatch(markup(container), excluded);
    assert.match(markup(container), included);
  }
});

test('each major section ends with sequential navigation driven by the same outline', () => {
  for (const group of toc) {
    const section = markup(byId(group.id));
    assert.match(section, new RegExp(`\\{@render sectionNav\\('${group.id}'\\)\\}\\s*</section>$`));
  }
  assert.match(source, /const previous = toc\[position - 1\]/);
  assert.match(source, /const next = toc\[position \+ 1\]/);
  assert.match(source, /Next: \{next.label\}/);
  assert.match(source, /Previous: \{previous.label\}/);
  assert.doesNotMatch(source, /class="section-direction"|Back to contents/, 'the existing sticky Contents bar is the mobile return path');
  assert.doesNotMatch(source, /class="quick-links"/);
});
