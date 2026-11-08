import test from 'node:test';
import assert from 'node:assert/strict';
import { mountAnalysisDisclosure } from '../src/lc0/analysisDisclosure.ts';

class Element extends EventTarget {
  hidden = false;
  attrs = {};
  children = [];
  constructor(doc) { super(); this.ownerDocument = doc; }
  contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
  setAttribute(key, value) { this.attrs[key] = value; }
  focus() { this.ownerDocument.activeElement = this; }
  closest() { return this; }
}
function fixture(enabled) {
  const doc = new EventTarget();
  const root = new Element(doc), trigger = new Element(doc), panel = new Element(doc);
  root.children = [trigger, panel];
  const controller = new AbortController();
  const disclosure = mountAnalysisDisclosure({ root, trigger, panel, signal: controller.signal, enabled });
  const escape = () => {
    const event = new Event('keydown', { cancelable: true });
    Object.defineProperty(event, 'key', { value: 'Escape' });
    doc.dispatchEvent(event);
    return event;
  };
  return { doc, root, trigger, panel, disclosure, controller, escape };
}
test('board tools and Copy use a disclosure with Escape focus restoration and outside close', () => {
  const { doc, root, trigger, panel, controller, escape } = fixture();
  assert.equal(panel.hidden, true);
  trigger.dispatchEvent(new Event('click'));
  assert.equal(panel.hidden, false);
  assert.equal(trigger.attrs['aria-expanded'], 'true');
  assert.equal(escape().defaultPrevented, true);
  assert.equal(panel.hidden, true);
  assert.equal(doc.activeElement, trigger);
  trigger.dispatchEvent(new Event('click'));
  doc.dispatchEvent(new Event('pointerdown'));
  assert.equal(panel.hidden, true);
  trigger.dispatchEvent(new Event('click'));
  const out = new Event('focusout');
  Object.defineProperty(out, 'relatedTarget', { value: null });
  root.dispatchEvent(out);
  assert.equal(panel.hidden, true);
  controller.abort();
  trigger.dispatchEvent(new Event('click'));
  assert.equal(panel.hidden, true, 'unmount removes listeners');
});
test('desktop keeps existing tools visible; resizing to mobile closes them', () => {
  let mobile = false;
  const { trigger, panel, disclosure } = fixture(() => mobile);
  assert.equal(panel.hidden, false);
  trigger.dispatchEvent(new Event('click'));
  assert.equal(trigger.attrs['aria-expanded'], 'false');
  mobile = true;
  disclosure.close();
  assert.equal(panel.hidden, true);
  trigger.dispatchEvent(new Event('click'));
  assert.equal(panel.hidden, false);
  mobile = false;
  disclosure.close();
  assert.equal(panel.hidden, false);
});
test('retry keeps the mobile Flip host visible even when focusout closes the disclosure', () => {
  let retry = false;
  const { root, panel, disclosure } = fixture(() => !retry);
  assert.equal(panel.hidden, true);
  retry = true;
  disclosure.close();
  assert.equal(panel.hidden, false);
  const out = new Event('focusout');
  Object.defineProperty(out, 'relatedTarget', { value: null });
  root.dispatchEvent(out);
  assert.equal(panel.hidden, false, 'closing must not hide the only Flip control during retry');
  retry = false;
  disclosure.close();
  assert.equal(panel.hidden, true);
});
test('a handled nested Escape does not also dismiss the parent disclosure', () => {
  const { doc, trigger, panel } = fixture();
  trigger.dispatchEvent(new Event('click'));
  const event = new Event('keydown', { cancelable: true });
  Object.defineProperty(event, 'key', { value: 'Escape' });
  event.preventDefault();
  doc.dispatchEvent(event);
  assert.equal(panel.hidden, false);
});
