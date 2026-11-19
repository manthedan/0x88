import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import test from 'node:test';
import ts from 'typescript';

const lookupNames = new Set(['el', 'maybeEl', 'inputEl', 'selectEl', 'buttonEl']);
function controllerContract(source) {
  const tree = ts.createSourceFile('controller.ts', source, ts.ScriptTarget.Latest, true);
  const lookups = new Set(),
    generatedIds = new Set();
  const addHtml = (text) => {
    for (const match of text.matchAll(/\bid\s*=\s*["']([\w-]+)["']/g)) generatedIds.add(match[1]);
  };
  function visit(node) {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && lookupNames.has(node.expression.text) && ts.isStringLiteralLike(node.arguments[0]))
      lookups.add(node.arguments[0].text);
    if (ts.isStringLiteralLike(node) || ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) addHtml(node.text);
    if (
      ts.isBinaryExpression(node) &&
      node.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
      ts.isPropertyAccessExpression(node.left) &&
      node.left.name.text === 'id' &&
      ts.isStringLiteralLike(node.right)
    )
      generatedIds.add(node.right.text);
    ts.forEachChild(node, visit);
  }
  visit(tree);
  return { lookups, generatedIds };
}

function markupIds(source, props = {}) {
  const ids = new Set();
  for (const match of source.matchAll(/\bid\s*=\s*["']([\w-]+)["']/g)) ids.add(match[1]);
  for (const match of source.matchAll(/\bid\s*=\s*\{`([^`]+)`\}/g)) {
    const value = match[1].replace(/\$\{(\w+)\}/g, (_all, key) => props[key] ?? '${' + key + '}');
    if (!value.includes('${')) ids.add(value);
  }
  return ids;
}

function routeIds(file, props = {}, seen = new Set()) {
  const key = JSON.stringify([file, props]);
  if (seen.has(key)) return new Set();
  seen.add(key);
  const source = readFileSync(file, 'utf8'),
    ids = markupIds(source, props);
  for (const imported of source.matchAll(/import\s+(\w+)\s+from\s+['"]([^'"]+\.svelte)['"]/g)) {
    const [, name, path] = imported;
    const child = /^[$#]lib\//.test(path) ? resolve('src/lib', path.slice(5)) : resolve(dirname(file), path);
    for (const use of source.matchAll(new RegExp(`<${name}\\b([^>]*?)/?>`, 'g'))) {
      const attributes = Object.fromEntries([...use[1].matchAll(/(\w+)=["']([^"']*)["']/g)].map((match) => [match[1], match[2]]));
      for (const id of routeIds(child, attributes, seen)) ids.add(id);
    }
  }
  return ids;
}

function missingIds(contract, available) {
  return [...contract.lookups].filter((id) => !available.has(id) && !contract.generatedIds.has(id)).sort();
}

for (const controller of ['play', 'analysis', 'arena'])
  test(`${controller}: every literal DOM lookup has route/component/generated markup`, () => {
    const contract = controllerContract(readFileSync(`src/lc0/${controller}Browser.ts`, 'utf8'));
    const available = routeIds(resolve(`src/routes/app/${controller}/+page.svelte`));
    assert.ok(contract.lookups.size > 30, 'the guard must actually scan the controller');
    assert.deepEqual(missingIds(contract, available), [], 'stale ids must be fixed, not silently ignored during teardown');
  });

test('the id contract resolves component props and generated HTML but rejects a new missing literal id', () => {
  const contract = controllerContract(`el('message'); inputEl('playRestoreNew'); maybeEl('generated'); el('typo'); const html='<div id="generated"></div>';`);
  const available = markupIds(`<p id="message"></p><button id={\`\${area}RestoreNew\`}></button>`, { area: 'play' });
  assert.deepEqual(missingIds(contract, available), ['typo']);
  assert.deepEqual(missingIds(controllerContract("el('analysisRestoreNew')"), available), ['analysisRestoreNew'], 'another route cannot satisfy the contract');
});
