---
name: code-style-simple
description: User's preference for simple, concise code without decorative comments
metadata:
  type: feedback
---

Code should be as straightforward, concise, and simple as possible. No special-character/separator comments (e.g. `# ===== section =====`), no verbose comments, and avoid unnecessary abstractions.

**Why:** User explicitly stated this preference while building rebuttal plotting scripts.
**How to apply:** Write lean functions, minimal comments (only where genuinely non-obvious), prefer plain straight-line code over layered helpers. For plots: no titles, no boldface, generally lowercase text (except acronyms), small figures.
