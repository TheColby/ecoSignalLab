## 2025-05-15 - [Navigation Accessibility and Active State Indication]
**Learning:** Visual verification of accessibility features like skip links can be automated using Playwright by programmatically focusing elements and checking their CSS properties (e.g., `top` or `visibility`) before and after focus transitions.
**Action:** Use `Path.resolve()` for comparing page paths in the documentation builder to ensure accurate `aria-current="page"` assignment across varied directory depths.
