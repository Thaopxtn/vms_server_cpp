# VMS Command Center UI Redesign

## Overall Mockup

The live monitoring screen is a single command-center viewport:

- 56px header with brand, server state, fullscreen, auto grid, notifications, user.
- Collapsible left navigation with live monitor, vehicles, persons, plates, events, statistics, settings.
- Compact control bar with status tabs, quality, AI actions, search, auto grid, and system metrics.
- Camera grid fills all remaining height and never paginates by default.
- Notification dock stays in the bottom-right corner and does not cover the camera grid center.

## Desktop 1920x1080

- Sidebar: 176px, topbar: 56px, toolbar: 56px.
- Grid uses all remaining pixels.
- <=64 cameras use normal mode with bottom controls.
- >64 cameras use thumbnail mode with dense gaps and hover controls.

## 4K

- Same layout engine, larger cell area.
- The grid recomputes on resize/fullscreen via `ResizeObserver`.
- 16:9 is used as the target ratio while still forcing all visible cameras into the viewport.

## Design System

- Background: `#08111F`
- Primary: `#0066FF`
- Success: `#00D26A`
- Warning: `#FFB020`
- Danger: `#FF4D4F`
- Radius: `10px`
- Font: Inter or system UI fallback
- Surfaces use dark translucent panels with light blur and restrained borders.

## Component Library

- `Topbar`: brand, server status, fullscreen, auto grid, operator.
- `SidebarNav`: icon plus text navigation, collapsible at narrow widths.
- `MonitorToolbar`: filter tabs, quality segmented control, AI controls, search, metrics.
- `CameraTile`: video frame, status dot, name, quality badge, snapshot, audio, record.
- `ThumbnailCameraTile`: dense variant for >64 cameras; controls appear on hover.
- `NotificationDock`: transient event messages, 5 second lifetime.
- `MetricPill`: compact CPU/RAM/GPU/AI counters.

## Auto Grid Rules

1. Filter cameras by search and status tab.
2. Sort recent AI alerts first, online cameras second.
3. Measure camera grid width and height.
4. Try every possible column count.
5. Choose the layout with the best 16:9 video ratio and largest useful tile area.
6. Fixed layouts `1x1` through `8x8` load only the current page of streams to avoid freezing the operator machine.
7. Auto layout is stream-safe and caps live iframes at 64 per page.
8. Spotlight layout loads 1 large camera plus up to 12 smaller surrounding cameras.
9. Recompute on resize, fullscreen, camera status refresh, search, and filter changes.

## User Flow

1. Operator opens the live monitor.
2. All cameras appear in one screen automatically.
3. Operator can search or filter to `Tat ca`, `AI`, `Offline`, or `Canh bao`.
4. AI events pulse the affected camera and move it toward the front.
5. Snapshot/record/audio controls stay visible in normal mode and appear on hover in thumbnail mode.
6. Alerts appear in the notification dock without blocking the camera grid.

## Wireframe

```text
+--------------------------------------------------------------------------+
| Logo VMS | Server online                         [Fullscreen] [Auto] [A] |
+----------+---------------------------------------------------------------+
| Sidebar  | Filter tabs | Quality | AI controls | Search | Metrics        |
|          +---------------------------------------------------------------+
|          |                                                               |
|          |  Auto-fit camera grid, no vertical camera scroll              |
|          |                                                               |
|          |  [Cam][Cam][Cam][Cam][Cam][Cam][Cam][Cam]                    |
|          |  [Cam][Cam][Cam][Cam][Cam][Cam][Cam][Cam]                    |
|          |  [Cam][Cam][Cam][Cam][Cam][Cam][Cam][Cam]                    |
|          |                                                               |
|          |                                      Notification dock        |
+----------+---------------------------------------------------------------+
```

## Figma-Ready Components

Use these frame names and variants:

- `Frame/CommandCenter/Desktop-1920`
- `Frame/CommandCenter/Desktop-4K`
- `Component/Topbar`
- `Component/SidebarNav`
- `Component/MonitorToolbar`
- `Component/CameraTile/Normal`
- `Component/CameraTile/Thumbnail`
- `Component/CameraTile/Offline`
- `Component/CameraTile/Alert`
- `Component/NotificationDock`
- `Component/MetricPill`

## Future React + Tailwind Structure

```text
src/
  app/
    App.tsx
    routes.tsx
  components/
    Topbar.tsx
    SidebarNav.tsx
    MonitorToolbar.tsx
    CameraGrid.tsx
    CameraTile.tsx
    NotificationDock.tsx
    MetricPill.tsx
  hooks/
    useAutoGrid.ts
    useCameraStatus.ts
    useGo2rtcStreams.ts
    useDetections.ts
  lib/
    api.ts
    cameraLayout.ts
    status.ts
  styles/
    tokens.ts
    globals.css
```

The current implementation keeps this design in `web/index.html` so the existing C++ server can serve it without a new build pipeline.
