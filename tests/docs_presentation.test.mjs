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

test('the outline and article agree on five peer sections and every topic has a heading', () => {
  const sectionIds = elements.filter(({ node }) => node.name === 'section').map(({ node }) => textAttr(node, 'id'));
  assert.deepEqual(sectionIds, ['pages', 'troubleshooting', 'engines', 'under-the-hood', 'licenses']);
  assert.deepEqual(toc.map((group) => group.id), sectionIds);
  for (const group of toc) {
    assert.equal(byId(group.id).children.find((node) => node.name === 'h2')?.name, 'h2');
    for (const item of group.items) {
      assert.equal(byId(item.id)?.name, 'h3', item.id);
      for (const child of item.children ?? []) assert.equal(byId(child.id)?.name, 'h4', child.id);
    }
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
});

test('mobile search is outside the collapsed panel and shares the desktop search focus action', () => {
  const trigger = elements.find(({ node }) => hasClass(node, 'mobile-search'));
  assert.ok(trigger);
  assert.ok(!trigger.ancestors.some((node) => textAttr(node, 'id') === 'toc-panel'));
  assert.match(markup(trigger.node), /Search docs/);
  assert.match(markup(trigger.node), /on:click=\{focusSearch\}/);
  assert.match(source, /if \(narrow\) tocOpen = true;\s+await tick\(\);\s+searchInput\?\.focus\(\)/);
  assert.match(source, /searchShortcut = '⌘K'/);
  assert.match(source, /let searchShortcut = 'Ctrl K'/);
});

test('all workspace launch links are with the title, ahead of the four-step procedure', () => {
  for (const workspace of ['analysis', 'arena', 'play']) {
    const heading = byId(`pages-${workspace}`);
    const card = elements.find(({ node, ancestors }) => node === heading && ancestors.some((parent) => hasClass(parent, 'page-block'))).ancestors.find((node) => hasClass(node, 'page-block'));
    const launch = descendants(card, 'a').find((node) => textAttr(node, 'href') === `/app/${workspace}/`);
    const steps = descendants(card, 'ol')[0];
    assert.ok(launch.end < steps.start);
    assert.equal(steps.children.filter((node) => node.name === 'li').length, 4);
    assert.ok(elements.find(({ node, ancestors }) => node === launch && ancestors.some((parent) => hasClass(parent, 'task-heading'))));
    assert.ok(attr(heading, 'aria-labelledby'), 'the heading name excludes its permalink');
  }
  const reference = elements.find(({ node }) => hasClass(node, 'task-reference')).node;
  assert.match(markup(reference), /<h4>Exploring the results<\/h4>/);
  const note = elements.find(({ node }) => hasClass(node, 'not-yet')).node;
  assert.ok(note.start > reference.end, 'the unchanged availability note stays separate');
});

test('subsection search containers exclude adjacent topics', () => {
  const containerNames = source.match(/const CONTAINERS = '([^']+)'/)[1].split(', ').map((name) => name.slice(1));
  for (const [id, className, excluded, included] of [
    ['cpu-wasm-relaxed-simd', 'runtime-subsection', /wasm32-wasip1/, /i32x4_relaxed_dot_i8x16_i7x16_add/],
    ['licenses-per-engine', 'source-links', /rights holders/, /corresponding-source\.tar\.gz/],
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
  assert.doesNotMatch(source, /class="quick-links"/);
});
