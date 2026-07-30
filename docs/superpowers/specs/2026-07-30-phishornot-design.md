# PhishOrNot — Design Spec

## Overview

Complete visual revamp of the PhishOrNot phishing detection web app with a **Bento Grid** design — dark, technical, and approachable. Full-width responsive layout with modular cards, subtle shadows, and staggered animations. Built with React 19 + Tailwind CSS v4 + React Router v7.

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

- **Bento UI** — subtle `shadow-sm` (`0 1px 3px rgba(0,0,0,0.3)`), `rounded-2xl` (16px) on cards
- **Card hover:** `hover:scale-[1.02]` with `hover:shadow-md` for lift effect
- **Button hover:** opacity shift (`hover:opacity-90`) with 150-200ms transition
- **Active states:** subtle scale (`active:scale-[0.98]`) for buttons
- **Focus states:** ring-2 with accent color for keyboard navigation
- **Borders:** `border border-border` on cards for definition

## Page Architecture

### 1. NavBar

- Dark surface (`#1E293B`) with bottom border
- Logo "phishornot?" in bold with shield icon
- Three nav tabs: Check (ShieldCheck icon), History (ClockCounterClockwise icon), Dashboard (ChartBar icon)
- Active tab: accent text + subtle active indicator
- Phosphor icons, no emojis
- Responsive: tabs collapse to icon-only on mobile (≤640px)

### 2. Check Page (Home `/`)

- **Full-width layout** (no max-width constraint, responsive px-6 padding)
- Hero section: centered heading + tagline + input (max-w-2lx)
- URL input: large, centered, with accent-colored CTA button
- Loading state: subtle spinner + "Analyzing..." text
- Result card: full-width bento tile, appears with fade-in + slide-up animation (200ms ease)
- Below result: small bento strip with quick stats / safety tips

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
- History items in **2-column bento grid** (responsive → 1-col on mobile)
- Each item: verdict badge, URL (truncated), confidence %, timestamp
- Each item clickable → navigates to Check page with that result loaded
- Empty state: "No checks yet" with appropriate icon
- Filter animation: items stagger in on load

### 5. Dashboard (`/dashboard`)

- **Full-width bento grid layout** (no max-width constraint, fills viewport with px-6 padding)
- Three stat cards: Total Checks, Phishing, Legitimate — in a responsive 3-column grid
- **Ratio bar:** spans full width, larger size (2x1 bento tile)
- **Recent Checks** and **Top Domains** in side-by-side 2-column grid (each 1x1 tile)
- Cards use `rounded-2xl`, `shadow-sm`, `hover:scale-[1.02]`, stagger entrance animation
- Empty state when no history exists

## Components

### Buttons

| Variant | Style |
|---------|-------|
| Primary (CTA) | Accent green bg, white text, rounded-xl, px-6 py-3, hover:opacity-90, active:scale-[0.98] |
| Secondary | Transparent, border, text muted, hover:text-white hover:border-white |
| Destructive | Red border, red text, hover:bg-red-900/20 |
| Ghost | No border, text muted, hover:text-white |

### Cards (Bento Tiles)

- Surface bg (`#1E293B`), border (`#475569`), **rounded-2xl** (16px)
- **Shadow:** `shadow-sm` (`0 1px 3px rgba(0,0,0,0.3)`)
- **Hover:** `hover:scale-[1.02] hover:shadow-md` with 200ms ease transition
- Padding: p-5 or p-6 depending on content density
- Stagger entrance via `motion-safe:animate-[fadeInSlideUp_300ms_ease]`

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
- **Bento grid stagger:** items fade-in + translate-y with 60ms staggered delay per item
- **Card hover:** `hover:scale-[1.02]` with 200ms ease
- **Loading spinner:** CSS `animate-spin`
- **Button press:** `active:scale-[0.98]`
- **Accordion expand:** rotate chevron 90°, max-height transition
- **`prefers-reduced-motion`:** all animations disabled via Tailwind's `motion-safe:` prefix

## Responsive Breakpoints

- Mobile: 375px+ (single column, icon-only nav, px-4)
- Tablet: 768px+ (2-column bento grid, px-6)
- Desktop: 1024px+ (3-column stats, full bento layout, px-8)
- Wide: 1440px+ (px-12, max card sizes capped)

## Anti-Patterns (Do Not) — from design system

- ❌ Emojis as icons (use Phosphor SVG)
- ❌ Narrow content on wide screens (full-width bento layout instead)
- ❌ Low contrast text (< 4.5:1)
- ❌ Instant state changes without transitions
- ❌ Missing `cursor-pointer` on clickable elements
- ❌ Invisible focus states
