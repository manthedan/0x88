/** Clipboard permissions/API availability must be reflected in the action's feedback. */
export async function writeClipboard(text: string, clipboard = globalThis.navigator?.clipboard): Promise<boolean> {
  try {
    if (!clipboard) return false;
    await clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
