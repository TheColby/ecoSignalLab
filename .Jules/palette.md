# Palette's Journal

## 2026-02-22 - [Documentation Accessibility & Interactivity]
**Learning:** Adding interactive features like "Copy to Clipboard" to static documentation generated via Python scripts requires careful state management for screen readers (using `aria-live="polite"`) and precise string escaping (doubling curly braces in f-strings). A "Skip to content" link is a fundamental but often overlooked accessibility win for documentation with long sidebars.
**Action:** Always include ARIA live regions for dynamic feedback and ensure landmark regions (nav, main) are correctly defined to support keyboard users.
