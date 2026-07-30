# PhishOrNot — Design Spec

## Overview

Complete visual revamp of the PhishOrNot phishing detection web app. Dark, technical, and approachable — targeting security-conscious general consumers. Built with React 19 + Tailwind CSS v4 + React Router v7.

## Design Tokens

### Color Palette

| Role | Hex | CSS Variable | Usage |
|------|-----|--------------|-------|
| Background | `#0F172A` | `--color-bg` | Page background |
| Surface | `#1E293B` | `--color-surface` | Cards, nav, panels |
| Surface Muted | `#272F42` | `--color-surface-muted` | Hover, subtle backgrounds |
| Foreground | `#F8FAFC` | `--color-text` | Primary text |
| Foreground Muted | `#94A3B8` | `--color-text-muted` | Secondary text, placeholders |
| Border | `#475569` | `--color-border` | Dividers, borders |
| Accent | `#22C55E` | `--color-accent` | CTAs, legitimate verdict, positive signals |
| Accent Glow | `rgba(34, 197, 94, 0.15)` | `--color-accent-glow` | Subtle accent glows |
| Destructive | `#EF4444` | `--color-destructive` | Phishing verdicts, errors, warnings |
| Destructive Glow | `rgba(239, 68, 68, 0.15)` | `--color-destructive-glow` | Subtle destructive glows |

### Typography

- **Font:** Inter (single family, all weights 300–700)
- **Headings:** `font-semibold`, tight `letter-spacing: -0.02em`
- **Body:** `font-normal`, `leading-relaxed`
- **Monospace:** `font-mono` for URLs and feature keys

### Spacing

Using Tailwind v4 default scale (`0.25rem` increments). Key tiers:
- `p-4` (16px) = card padding
- `p-6` (24px) = large card padding
- `gap-3` (12px) = component gap
- `space-y-6` (24px) = section gap

### Effects

- **Flat design** — no shadows or gradients
- **Hover states:** opacity shift (`hover:opacity-90`) or color shift with 150-200ms transition
- **Active states:** subtle scale (`active:scale-[0.98]`) for buttons
- **Focus states:** ring-2 with accent color for keyboard navigation
- **Borders:** `border` on cards/surfaces, not shadows

## Page Architecture

### 1. NavBar

- Dark surface (`#1E293B`) with bottom border
- Logo "phishornot?" in bold with shield icon
- Three nav tabs: Check (ShieldCheck icon), History (ClockCounterClockwise icon), Dashboard (ChartBar icon)
- Active tab: accent text + subtle active indicator
- Phosphor icons, no emojis
- Responsive: tabs collapse to icon-only on mobile (≤640px)

### 2. Check Page (Home `/`)

- Hero-style layout: centered, large typography
- Tagline: oversized heading + short descriptive text
- URL input: large, centered, with accent-colored CTA button
- Loading state: subtle spinner + "Analyzing..." text with staggered dots animation
- Result card: appears with fade-in + slide-up animation (200ms ease)
- Empty state: descriptive placeholder text, centered

### 3. ResultCard

- Phishing verdict: red badge with dot indicator, red-tinted card border
- Legitimate verdict: green badge with dot indicator, green-tinted card border
- Confidence bar: gradient fill with tick marks, dynamic color based on verdict
- URL display: monospace, truncated with `break-all`
- "Why this verdict?" section: bullet list with colored dots (red/green per reason)
- Technical details: collapsible accordion with chevron animation
- Share button: link icon, copies shareable URL to clipboard

### 4. History Panel (`/history`)

- Search input with magnifying glass icon
- "Clear All" button (destructive style)
- List of history items: verdict badge, URL (truncated), confidence %, timestamp
- Each item clickable → navigates to Check page with that result loaded
- Empty state: "No checks yet" with appropriate icon
- Filter animation: items stagger in on load

### 5. Dashboard (`/dashboard`)

- Three stat cards: Total Checks, Phishing, Legitimate — in a responsive grid
- Ratio bar: split red/green bar with percentage labels
- Recent Checks list (last 10)
- Most Checked Domains list (top 10)
- Empty state when no history exists
- Cards with consistent surface styling, border-based separation

## Components

### Buttons

| Variant | Style |
|---------|-------|
| Primary (CTA) | Accent green bg, white text, rounded-xl, px-6 py-3, hover:opacity-90, active:scale-[0.98] |
| Secondary | Transparent, border, text muted, hover:text-white hover:border-white |
| Destructive | Red border, red text, hover:bg-red-900/20 |
| Ghost | No border, text muted, hover:text-white |

### Cards

- Surface bg (`#1E293B`), border (`#475569`), rounded-xl
- Padding: p-5 or p-6 depending on content density
- No hover elevation change (flat design)

### Inputs

- Surface bg, border, rounded-xl
- Focus: ring-2 ring-accent/30, border-accent
- Placeholder: muted text

### Badges

- Phishing: bg-red-900/30, text-red-400, border-red-800, dot indicator
- Legitimate: bg-green-900/30, text-green-400, border-green-800, dot indicator

## Icons

Using `@phosphor-icons/react` (regular weight, 20px default):

| Location | Icon |
|----------|------|
| Nav: Check | `ShieldCheck` |
| Nav: History | `ClockCounterClockwise` |
| Nav: Dashboard | `ChartBar` |
| Share button | `ShareNetwork` |
| Search | `MagnifyingGlass` |
| Collapse/Expand | `CaretRight` (rotates) |
| Phishing alert | `WarningCircle` |
| Safe | `ShieldCheck` |
| Empty states | `ShieldSlash` |

## Animation

- **Page transitions:** 150-200ms ease, no GSAP (keep it lightweight)
- **Result appearance:** fade-in + translate-y(8px) → 0, 200ms ease
- **Loading spinner:** CSS `animate-spin`
- **Button press:** `active:scale-[0.98]`
- **Accordion expand:** rotate chevron 90°, max-height transition
- **`prefers-reduced-motion`:** all animations disabled via Tailwind's `motion-safe:` prefix

## Responsive Breakpoints

- Mobile: 375px+ (single column, icon-only nav)
- Tablet: 768px+ (two-column dashboard)
- Desktop: 1024px+ (max-w-3xl content width)
- Wide: 1440px+

## Anti-Patterns (Do Not) — from design system

- ❌ Emojis as icons (use Phosphor SVG)
- ❌ Layout-shifting hover transforms
- ❌ Low contrast text (< 4.5:1)
- ❌ Instant state changes without transitions
- ❌ Missing `cursor-pointer` on clickable elements
- ❌ Invisible focus states
