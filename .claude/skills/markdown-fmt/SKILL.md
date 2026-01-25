---
name: markdown-fmt
description: Format and clean up markdown files. Use when editing documentation or README files.
---

# Markdown Formatting

When invoked, clean up and format markdown files consistently.

## Formatting Rules

### Headings
```markdown
# H1 - Document title (only one per file)
## H2 - Major sections
### H3 - Subsections
#### H4 - Minor subsections (use sparingly)
```

- One blank line before and after headings
- No trailing punctuation
- Sentence case (not Title Case for all words)

### Lists
```markdown
Unordered:
- Item one
- Item two
  - Nested item
  - Another nested

Ordered:
1. First step
2. Second step
3. Third step

Task lists:
- [ ] Incomplete task
- [x] Completed task
```

### Code
````markdown
Inline: Use `backticks` for code.

Block:
```typescript
const example = 'code block';
```

Specify language for syntax highlighting.
````

### Links
```markdown
[Link text](https://example.com)
[Relative link](./other-file.md)
[Reference style][ref]

[ref]: https://example.com
```

### Images
```markdown
![Alt text](image.png)
![Alt text](image.png "Optional title")
```

### Tables
```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |

Alignment:
| Left | Center | Right |
|:-----|:------:|------:|
| L    |   C    |     R |
```

### Blockquotes
```markdown
> Single line quote

> Multi-line quote
> continues here
>
> With paragraph break
```

### Horizontal Rules
```markdown
---

(Three dashes, with blank lines before and after)
```

## Common Fixes

### Inconsistent heading levels
```markdown
# Bad - Skipping levels
# Title
### Subsection (skipped H2!)

# Good - Sequential levels
# Title
## Section
### Subsection
```

### Trailing whitespace
Remove spaces at end of lines (except for intentional line breaks).

### Missing blank lines
```markdown
# Bad - No spacing
## Heading
Content right after.
## Next heading

# Good - Proper spacing
## Heading

Content with space.

## Next heading
```

### Long lines
Break lines at ~80-100 characters for readability in source.
(Rendered output will flow normally.)

### Inconsistent list markers
```markdown
# Bad - Mixed markers
- Item one
* Item two
+ Item three

# Good - Consistent
- Item one
- Item two
- Item three
```

## Structure Template

```markdown
# Document Title

Brief introduction paragraph.

## Section 1

Content for section 1.

### Subsection 1.1

More detailed content.

## Section 2

Content for section 2.

## See Also

- [Related document](./related.md)
- [External resource](https://example.com)
```

## Linting

```bash
# Lint with markdownlint
npx markdownlint-cli "**/*.md"

# Fix automatically
npx markdownlint-cli --fix "**/*.md"

# Prettier for formatting
npx prettier --write "**/*.md"
```

## Checklist

- [ ] Single H1 at top
- [ ] Sequential heading levels
- [ ] Blank lines around headings and blocks
- [ ] Consistent list markers
- [ ] Code blocks have language specified
- [ ] Links work (no broken links)
- [ ] Tables aligned properly
- [ ] No trailing whitespace
