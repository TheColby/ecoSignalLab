## 2025-05-14 - [Documentation Accessibility Overhaul]
**Learning:** Documentation generated from Markdown often neglects basic keyboard navigation and screen reader landmarks. Adding a "Skip to content" link and semantic navigation landmarks significantly improves the experience for power users and users with disabilities without cluttering the visual UI.
**Action:** Always include a visually-hidden-until-focused skip link and use `aria-current` to indicate the active page in navigation menus.
