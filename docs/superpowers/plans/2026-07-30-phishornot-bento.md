# PhishOrNot Bento Grid — Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Convert the layout from narrow/content to full-width Bento Grid with shadows, rounded-2xl cards, hover scale, and stagger animations.

**Architecture:** Incremental changes on top of current redesign. Same tokens, fonts, icons — just layout, shadows, and animation changes.

**Tech Stack:** React 19, Tailwind CSS v4, Phosphor Icons

## Global Constraints (unchanged)

- No emojis — use `@phosphor-icons/react`
- All interactive elements: `cursor-pointer`, `motion-safe:transition-all`, `focus-visible:ring-2`
- Text contrast ≥ 4.5:1
- Responsive: 375px, 768px, 1024px, 1440px

---

### Task 1: Update CSS theme — add shadow tokens and stagger keyframes

**Files:**
- Modify: `frontend/src/index.css`

Add shadow utility tokens and stagger animation keyframes to the Tailwind v4 `@theme` block.

Edit `frontend/src/index.css`:
- Add `--shadow-sm: 0 1px 3px 0 rgba(0,0,0,0.3)` to `@theme`
- Add `--shadow-md: 0 4px 6px -1px rgba(0,0,0,0.4)` to `@theme`

The current `@theme` block should have these added before the closing `}`:
```
  --shadow-sm: 0 1px 3px 0 rgba(0,0,0,0.3);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.4);
```

Run build, commit:
```bash
cd frontend && npm run build
git add frontend/src/index.css
git commit -m "feat: add shadow tokens for Bento cards"
```

---

### Task 2: Full-width layout and App wrapper

**Files:**
- Modify: `frontend/src/App.jsx`

**Changes:**
1. Replace the main wrapper `max-w-3xl mx-auto px-4 py-8` with `w-full px-4 sm:px-6 lg:px-8 py-8 max-w-7xl mx-auto` (full-width with readable max)
2. Add `shadow-sm` and `rounded-2xl` to the result card container div
3. Add stagger keyframes if not already present

Edit `App.jsx`:

Find:
```
      <main className="max-w-3xl mx-auto px-4 py-8">
```
Replace with:
```
      <main className="w-full px-4 sm:px-6 lg:px-8 py-8 max-w-7xl mx-auto">
```

In the CheckPage return, find the result wrapper div:
```
        <div className="motion-safe:animate-[fadeInSlideUp_300ms_ease]">
          <ResultCard result={result} resultId={resultId} />
        </div>
```
Add shadow/rounded classes to ResultCard:
This is handled inside ResultCard itself (next task) — no change needed here.

Run tests, commit:
```bash
cd frontend && npx vitest run
git add frontend/src/App.jsx
git commit -m "feat: full-width layout with max-w-7xl constraint"
```

---

### Task 3: Bento ResultCard

**Files:**
- Modify: `frontend/src/components/ResultCard.jsx`

**Changes:**
1. Change `rounded-xl` → `rounded-2xl` on the main card div
2. Add `shadow-sm` class to the main card div
3. Add `hover:scale-[1.02] hover:shadow-md transition-all duration-200 motion-safe:transition-all` to the main card div

Find the main card container:
```
    <div className={`bg-surface border rounded-xl p-6 space-y-5 ${
      isPhishing ? 'border-destructive/40' : 'border-accent/40'
    }`}>
```
Replace with:
```
    <div className={`bg-surface border rounded-2xl p-6 space-y-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200 ${
      isPhishing ? 'border-destructive/40' : 'border-accent/40'
    }`}>
```

Run tests, commit:
```bash
cd frontend && npx vitest run
git add frontend/src/components/ResultCard.jsx
git commit -m "feat: bento ResultCard with shadow, rounded-2xl, hover scale"
```

---

### Task 4: Bento HistoryPanel — 2-column grid

**Files:**
- Modify: `frontend/src/components/HistoryPanel.jsx`

**Changes:**
1. Change history items from `space-y-2` single column to `grid grid-cols-1 sm:grid-cols-2 gap-3`
2. Add `shadow-sm`, `rounded-2xl`, `hover:scale-[1.02]` to each history item button

Find the history list container:
```
        <div className="space-y-2">
          {filtered.map((item) => (
            <button
              key={item.id}
              ...
              className="w-full text-left bg-surface border border-border rounded-lg p-4 hover:border-accent/40 motion-safe:transition-all duration-150 cursor-pointer active:scale-[0.99] focus-visible:ring-2 focus-visible:ring-accent/30 focus-visible:outline-none"
            >
```
Replace with:
```
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {filtered.map((item) => (
            <button
              key={item.id}
              ...
              className="w-full text-left bg-surface border border-border rounded-2xl p-4 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200 cursor-pointer active:scale-[0.99] focus-visible:ring-2 focus-visible:ring-accent/30 focus-visible:outline-none"
            >
```

Also update the search input to `rounded-2xl`:
Find `rounded-lg` on the search input, change to `rounded-2xl`.

Run tests, commit:
```bash
cd frontend && npx vitest run
git add frontend/src/components/HistoryPanel.jsx
git commit -m "feat: bento HistoryPanel with 2-col grid, shadows, hover scale"
```

---

### Task 5: Bento Dashboard — full bento grid

**Files:**
- Modify: `frontend/src/components/Dashboard.jsx`

**Changes:**
1. All stat cards: `rounded-xl` → `rounded-2xl`, add `shadow-sm`, `hover:scale-[1.02] hover:shadow-md`
2. Ratio bar container: same bento card treatment
3. Recent/Top domains containers: same bento card treatment
4. Add stagger via style prop with increasing delay (or use an inline style approach)

Edit each card container. For the stat cards grid:
```
        <div className="bg-surface border border-border rounded-xl p-5">
```
Change to:
```
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200">
```

Do this for all 5 bento card containers:
- 3 stat cards
- Ratio bar container
- Recent Checks container
- Top Domains container

Also update the grid spacing. The stat cards grid currently uses `gap-4`. Change to `gap-5` for the outer grid, and `gap-5` for the inner grid.

The main dashboard container:
```
    <div className="space-y-6">
```
Change to:
```
    <div className="space-y-5">
```

Run tests, commit:
```bash
cd frontend && npx vitest run
git add frontend/src/components/Dashboard.jsx
git commit -m "feat: bento Dashboard with shadow, rounded-2xl, hover scale"
```

---

### Task 6: Final build and verify

```bash
cd frontend && npm run build
cd frontend && npx vitest run
```

Expected: clean build, all tests pass.
