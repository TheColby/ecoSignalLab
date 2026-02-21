## 2025-05-14 - [Documentation Interactivity]
**Learning:** When adding "Copy to Clipboard" buttons to code blocks, avoid using `innerText` on the parent container if the button is appended as a child, as it will include the button's own text in the clipboard content. Always target the specific code element or clone and prune the DOM before extraction. Additionally, use `aria-live="polite"` to provide non-visual feedback for state changes (e.g., "Copied!").
**Action:** Use specific selectors for content extraction and ensure clipboard APIs are checked for availability in non-secure contexts.
