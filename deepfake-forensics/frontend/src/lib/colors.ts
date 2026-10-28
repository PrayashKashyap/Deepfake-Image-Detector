// Validated reference palette (dataviz skill, references/palette.md). Categorical slots are used in
// fixed order; REAL/FAKE always map to slots 1/2 so identity never changes between charts.
export const SERIES_LIGHT = ['#2a78d6', '#eb6834', '#1baf7a'] as const
export const SERIES_DARK = ['#3987e5', '#d95926', '#199e70'] as const

export function seriesColors(dark: boolean): readonly string[] {
  return dark ? SERIES_DARK : SERIES_LIGHT
}

export function chrome(dark: boolean) {
  return dark
    ? { grid: '#2c2c2a', axis: '#383835', muted: '#898781', text: '#c3c2b7', surface: '#1a1a19' }
    : { grid: '#e1e0d9', axis: '#c3c2b7', muted: '#898781', text: '#52514e', surface: '#fcfcfb' }
}

/** Sequential blue ramp for magnitude (confusion-matrix cells). */
export const SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
