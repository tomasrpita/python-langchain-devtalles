---
description: Instruct the AI to generate Git commit messages following the Conventional Commits specification.

applyTo: "**"

---

# Git Commit Message Guidelines

Whenever you generate a Git commit message, always follow the Conventional Commits specification.

## Format

```
<type>(<optional-scope>): <short description>
```

### Allowed types

- feat
- fix
- docs
- style
- refactor
- perf
- test
- build
- ci
- chore
- revert

## Rules

- Use the imperative mood (e.g. "add", "fix", "remove", never "added" or "fixed").
- Write the summary in lowercase, except for proper nouns.
- Do not end the summary with a period.
- Keep the first line under 72 characters.
- Only include a scope when it adds useful context.
- Explain **what** changed, not **how** it was implemented.
- If a breaking change is introduced, append `!` after the type or scope and include a `BREAKING CHANGE:` section in the body.
- If additional explanation is useful, leave a blank line after the title and write a concise body.

## Examples

```
feat(auth): add OAuth login support

fix(api): handle null response from OpenAI

refactor(database): simplify connection management

docs: update installation instructions

test(parser): add edge case coverage

chore: update dependencies
```