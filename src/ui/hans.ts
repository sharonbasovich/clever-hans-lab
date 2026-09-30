// Original SVG art of Clever Hans the horse (drawn for this project).
export function hansSvg(cls = ''): string {
  return `
<svg viewBox="0 0 200 160" class="hans ${cls}" role="img" aria-label="Clever Hans, a cartoon horse, in profile">
  <defs>
    <linearGradient id="coat" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#b07a45"/>
      <stop offset="1" stop-color="#8a5a2b"/>
    </linearGradient>
  </defs>
  <!-- body -->
  <ellipse cx="105" cy="102" rx="58" ry="34" fill="url(#coat)"/>
  <!-- neck + head -->
  <path d="M52 88 Q40 50 52 34 Q58 26 70 28 Q92 32 96 52 Q99 66 92 78 L70 96 Z" fill="url(#coat)"/>
  <!-- muzzle -->
  <path d="M52 34 Q40 38 36 48 Q33 57 42 60 Q52 62 60 56 L66 44 Z" fill="#a5713d"/>
  <!-- ear -->
  <path d="M72 28 L80 12 L86 30 Z" fill="#8a5a2b"/>
  <!-- mane -->
  <path d="M70 30 Q86 44 84 64 Q94 50 90 34 Q84 24 74 24 Z" fill="#5a381a"/>
  <!-- eye -->
  <circle cx="62" cy="42" r="4" fill="#1c1917"/>
  <circle cx="63.4" cy="40.6" r="1.4" fill="#fff"/>
  <!-- nostril -->
  <ellipse cx="42" cy="52" rx="2.4" ry="1.6" fill="#3f2a12"/>
  <!-- legs -->
  <rect x="66" y="120" width="10" height="34" rx="4" fill="#8a5a2b"/>
  <rect x="96" y="122" width="10" height="32" rx="4" fill="#8a5a2b"/>
  <rect x="128" y="120" width="10" height="34" rx="4" fill="#8a5a2b"/>
  <rect x="150" y="118" width="10" height="36" rx="4" fill="#8a5a2b"/>
  <!-- hooves -->
  <rect x="64" y="150" width="14" height="8" rx="3" fill="#3f2a12"/>
  <rect x="94" y="150" width="14" height="8" rx="3" fill="#3f2a12"/>
  <rect x="126" y="150" width="14" height="8" rx="3" fill="#3f2a12"/>
  <rect x="148" y="150" width="14" height="8" rx="3" fill="#3f2a12"/>
  <!-- tail -->
  <path d="M160 84 Q184 92 178 118 Q172 132 166 122 Q176 106 158 96 Z" fill="#5a381a"/>
</svg>`;
}
