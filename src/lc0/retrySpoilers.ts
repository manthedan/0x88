/**
 * Keep the real panel nodes (and listeners) off-document, not merely aria-hidden.
 * Empty id-only skeletons let late lifecycle callbacks find their hosts without
 * retaining SAN, scores, probabilities, classification labels or data-PVs.
 * Callers must also suspend answer-producing renderers while the vault is open.
 */
export function quarantineRetryPanels(hosts: HTMLElement[]): () => void {
  const saved = hosts.map((host) => {
    const nodes = [...host.childNodes];
    const hidden = host.hidden;
    const placeholders = [...host.querySelectorAll<HTMLElement>('[id]')].map((element) => {
      const placeholder = host.ownerDocument.createElement(element.tagName.toLowerCase());
      placeholder.id = element.id;
      placeholder.hidden = true;
      return placeholder;
    });
    host.replaceChildren(...placeholders);
    host.hidden = true;
    return { host, nodes, hidden };
  });
  return () => {
    for (const { host, nodes, hidden } of saved) {
      host.replaceChildren(...nodes);
      host.hidden = hidden;
    }
  };
}
