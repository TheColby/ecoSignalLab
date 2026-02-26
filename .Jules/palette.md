## 2025-05-14 - Documentation Accessibility and Navigation UX
**Learning:** Custom documentation generators often lack basic accessibility features like "Skip to content" links and proper ARIA landmarks. Additionally, providing visual feedback for the current page in the navigation (using `aria-current="page"`) significantly improves orientation in large documentation sets.
**Action:** Always include a skip link and `aria-current` in documentation templates. Use `Path.resolve()` when comparing paths in Python to handle relative path variations accurately.
