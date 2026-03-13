import { useState, useRef, useEffect } from "react";

const BLOCKS = [{"speaker":"SPEAKER_04","start":2.9,"end":18.82},{"speaker":"SPEAKER_02","start":18.84,"end":19.84},{"speaker":"SPEAKER_02","start":21.864,"end":200.442},{"speaker":"SPEAKER_04","start":202.548,"end":209.697},{"speaker":"SPEAKER_02","start":214.764,"end":246.114},{"speaker":"SPEAKER_04","start":248.579,"end":271.737},{"speaker":"SPEAKER_02","start":273.299,"end":274.461},{"speaker":"SPEAKER_04","start":274.481,"end":297.001},{"speaker":"SPEAKER_02","start":298.243,"end":303.109},{"speaker":"SPEAKER_02","start":305.182,"end":334.48},{"speaker":"SPEAKER_04","start":335.489,"end":349.063},{"speaker":"SPEAKER_02","start":350.564,"end":379.47},{"speaker":"SPEAKER_02","start":383.214,"end":397.804},{"speaker":"SPEAKER_01","start":398.526,"end":398.866},{"speaker":"SPEAKER_04","start":399.828,"end":407.464},{"speaker":"SPEAKER_02","start":407.484,"end":412.354},{"speaker":"SPEAKER_04","start":412.675,"end":433.163},{"speaker":"SPEAKER_03","start":436.548,"end":553.078},{"speaker":"SPEAKER_02","start":556.248,"end":564.557},{"speaker":"SPEAKER_03","start":565.578,"end":565.738},{"speaker":"SPEAKER_02","start":566.9,"end":574.508},{"speaker":"SPEAKER_02","start":576.65,"end":577.73},{"speaker":"SPEAKER_04","start":581.678,"end":602.011},{"speaker":"SPEAKER_02","start":603.954,"end":659.273},{"speaker":"SPEAKER_03","start":661.042,"end":697.676},{"speaker":"SPEAKER_02","start":700.319,"end":721.162},{"speaker":"SPEAKER_03","start":722.684,"end":724.326},{"speaker":"SPEAKER_02","start":725.027,"end":758.363},{"speaker":"SPEAKER_04","start":761.92,"end":769.747},{"speaker":"SPEAKER_00","start":771.008,"end":829.491},{"speaker":"SPEAKER_04","start":831.412,"end":858.853},{"speaker":"SPEAKER_02","start":860.605,"end":896.774},{"speaker":"SPEAKER_00","start":901.601,"end":921.639},{"speaker":"SPEAKER_02","start":922.48,"end":925.123},{"speaker":"SPEAKER_00","start":926.044,"end":942.081},{"speaker":"SPEAKER_02","start":942.871,"end":999.392},{"speaker":"SPEAKER_02","start":1001.63,"end":1009.018},{"speaker":"SPEAKER_02","start":1011.6,"end":1020.75},{"speaker":"SPEAKER_01","start":1024.334,"end":1030.54},{"speaker":"SPEAKER_02","start":1030.874,"end":1040.931},{"speaker":"SPEAKER_02","start":1044.617,"end":1046.84},{"speaker":"SPEAKER_01","start":1047.401,"end":1063.771},{"speaker":"SPEAKER_02","start":1070.037,"end":1074.241},{"speaker":"SPEAKER_04","start":1086.633,"end":1110.898},{"speaker":"SPEAKER_02","start":1118.995,"end":1129.23},{"speaker":"SPEAKER_02","start":1132.354,"end":1138.523},{"speaker":"SPEAKER_02","start":1147.261,"end":1203.593},{"speaker":"SPEAKER_02","start":1206.222,"end":1218.574},{"speaker":"SPEAKER_00","start":1219.034,"end":1267.824},{"speaker":"SPEAKER_02","start":1270.511,"end":1272.496},{"speaker":"SPEAKER_04","start":1272.797,"end":1273.977},{"speaker":"SPEAKER_04","start":1277.502,"end":1295.044},{"speaker":"SPEAKER_02","start":1297.747,"end":1306.378},{"speaker":"SPEAKER_04","start":1310.29,"end":1322.305},{"speaker":"SPEAKER_02","start":1322.325,"end":1334.661},{"speaker":"SPEAKER_03","start":1339.535,"end":1352.774},{"speaker":"SPEAKER_02","start":1353.836,"end":1398.534},{"speaker":"SPEAKER_03","start":1399.516,"end":1407.933},{"speaker":"SPEAKER_02","start":1407.953,"end":1409.316},{"speaker":"SPEAKER_04","start":1443.856,"end":1457.84},{"speaker":"SPEAKER_02","start":1462.588,"end":1463.89},{"speaker":"SPEAKER_04","start":1467.295,"end":1485.743},{"speaker":"SPEAKER_04","start":1490.768,"end":1506.629},{"speaker":"SPEAKER_02","start":1508.132,"end":1508.492},{"speaker":"SPEAKER_04","start":1510.515,"end":1510.735}];

const SPEAKERS = ["SPEAKER_00","SPEAKER_01","SPEAKER_02","SPEAKER_03","SPEAKER_04"];

const SPEAKER_LABELS = {
  SPEAKER_00: "Speaker A",
  SPEAKER_01: "Speaker B",
  SPEAKER_02: "Speaker C",
  SPEAKER_03: "Speaker D",
  SPEAKER_04: "Speaker E",
};

const COLORS = {
  SPEAKER_00: { bar: "#0ea5e9", glow: "#38bdf8", bg: "rgba(14,165,233,0.12)" },
  SPEAKER_01: { bar: "#a78bfa", glow: "#c4b5fd", bg: "rgba(167,139,250,0.12)" },
  SPEAKER_02: { bar: "#34d399", glow: "#6ee7b7", bg: "rgba(52,211,153,0.12)" },
  SPEAKER_03: { bar: "#f97316", glow: "#fb923c", bg: "rgba(249,115,22,0.12)" },
  SPEAKER_04: { bar: "#f43f5e", glow: "#fb7185", bg: "rgba(244,63,94,0.12)" },
};

const TOTAL_START = 0;
const TOTAL_END = 1510.735;
const DURATION = TOTAL_END - TOTAL_START;

function fmtTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function fmtDuration(secs) {
  if (secs < 60) return `${secs.toFixed(1)}s`;
  return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
}

export default function MeetingGantt() {
  const [tooltip, setTooltip] = useState(null);
  const [hoveredSpeaker, setHoveredSpeaker] = useState(null);
  const [zoom, setZoom] = useState({ start: 0, end: DURATION });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const chartRef = useRef(null);

  const viewDur = zoom.end - zoom.start;

  const speakerStats = SPEAKERS.map((sp) => {
    const spBlocks = BLOCKS.filter((b) => b.speaker === sp);
    const totalTime = spBlocks.reduce((acc, b) => acc + (b.end - b.start), 0);
    return { speaker: sp, totalTime, blockCount: spBlocks.length };
  });
  const maxTime = Math.max(...speakerStats.map((s) => s.totalTime));

  const ticks = [];
  const tickCount = 8;
  for (let i = 0; i <= tickCount; i++) {
    ticks.push(zoom.start + (viewDur * i) / tickCount);
  }

  const handleMouseMove = (e, block) => {
    setTooltip({
      x: e.clientX,
      y: e.clientY,
      block,
    });
  };

  const handleChartMouseDown = (e) => {
    if (!chartRef.current) return;
    const rect = chartRef.current.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const t = zoom.start + pct * viewDur;
    setIsDragging(true);
    setDragStart(t);
  };

  const handleChartMouseUp = (e) => {
    if (!isDragging || dragStart === null) return;
    if (!chartRef.current) return;
    const rect = chartRef.current.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const t = zoom.start + pct * viewDur;
    const lo = Math.max(0, Math.min(dragStart, t));
    const hi = Math.min(DURATION, Math.max(dragStart, t));
    if (hi - lo > 5) {
      setZoom({ start: lo, end: hi });
    }
    setIsDragging(false);
    setDragStart(null);
  };

  const resetZoom = () => setZoom({ start: 0, end: DURATION });

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0a0f1e 0%, #0d1526 50%, #0a0f1e 100%)",
      fontFamily: "'IBM Plex Mono', monospace",
      color: "#e2e8f0",
      padding: "0",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@400;600;700&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0a0f1e; }
        ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
        .gantt-bar {
          transition: filter 0.15s, opacity 0.15s;
          cursor: pointer;
          border-radius: 3px;
        }
        .gantt-bar:hover { filter: brightness(1.3); }
        .speaker-row {
          transition: background 0.2s;
        }
        .speaker-row:hover {
          background: rgba(255,255,255,0.03);
        }
        .stat-bar-fill {
          transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1);
        }
        .tick-line {
          opacity: 0.15;
          stroke-dasharray: 4 4;
        }
        .zoom-hint { animation: fadeIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
      `}</style>

      {/* Header */}
      <div style={{
        padding: "32px 48px 24px",
        borderBottom: "1px solid rgba(255,255,255,0.07)",
        display: "flex",
        alignItems: "flex-end",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: "16px",
      }}>
        <div>
          <div style={{ fontSize: "10px", letterSpacing: "0.3em", color: "#64748b", marginBottom: "6px", textTransform: "uppercase" }}>
            Automated Meeting Analysis
          </div>
          <h1 style={{
            margin: 0,
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: "26px",
            fontWeight: 700,
            letterSpacing: "-0.02em",
            background: "linear-gradient(90deg, #e2e8f0 0%, #94a3b8 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>
            Speaking Time · Gantt View
          </h1>
          <div style={{ fontSize: "11px", color: "#475569", marginTop: "4px" }}>
            DesignPairSession_03-30-2023 &nbsp;·&nbsp; {SPEAKERS.length} speakers &nbsp;·&nbsp; {fmtDuration(DURATION)} total
          </div>
        </div>
        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          {zoom.start !== 0 || zoom.end !== DURATION ? (
            <button
              onClick={resetZoom}
              className="zoom-hint"
              style={{
                background: "rgba(255,255,255,0.08)",
                border: "1px solid rgba(255,255,255,0.15)",
                borderRadius: "6px",
                color: "#94a3b8",
                fontSize: "11px",
                padding: "6px 14px",
                cursor: "pointer",
                letterSpacing: "0.05em",
                fontFamily: "'IBM Plex Mono', monospace",
              }}
            >
              ← Reset Zoom
            </button>
          ) : (
            <div style={{ fontSize: "10px", color: "#334155", letterSpacing: "0.05em" }}>
              Drag on chart to zoom
            </div>
          )}
        </div>
      </div>

      <div style={{ display: "flex", gap: 0 }}>
        {/* Left: Stats Panel */}
        <div style={{
          width: "220px",
          flexShrink: 0,
          padding: "32px 24px",
          borderRight: "1px solid rgba(255,255,255,0.06)",
        }}>
          <div style={{ fontSize: "9px", letterSpacing: "0.25em", color: "#475569", marginBottom: "20px", textTransform: "uppercase" }}>
            Speaking Share
          </div>
          {speakerStats.map(({ speaker, totalTime, blockCount }) => {
            const pct = (totalTime / maxTime) * 100;
            const col = COLORS[speaker];
            const isHovered = hoveredSpeaker === speaker;
            return (
              <div
                key={speaker}
                style={{ marginBottom: "18px", cursor: "pointer", opacity: hoveredSpeaker && !isHovered ? 0.4 : 1, transition: "opacity 0.2s" }}
                onMouseEnter={() => setHoveredSpeaker(speaker)}
                onMouseLeave={() => setHoveredSpeaker(null)}
              >
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "5px", alignItems: "center" }}>
                  <span style={{ fontSize: "11px", fontWeight: 500, color: isHovered ? col.glow : "#94a3b8" }}>
                    {SPEAKER_LABELS[speaker]}
                  </span>
                  <span style={{ fontSize: "10px", color: "#475569" }}>{fmtDuration(totalTime)}</span>
                </div>
                <div style={{ height: "5px", background: "rgba(255,255,255,0.06)", borderRadius: "3px", overflow: "hidden" }}>
                  <div
                    className="stat-bar-fill"
                    style={{
                      height: "100%",
                      width: `${pct}%`,
                      background: `linear-gradient(90deg, ${col.bar}, ${col.glow})`,
                      borderRadius: "3px",
                      boxShadow: isHovered ? `0 0 8px ${col.glow}` : "none",
                    }}
                  />
                </div>
                <div style={{ fontSize: "9px", color: "#334155", marginTop: "3px" }}>
                  {blockCount} segment{blockCount !== 1 ? "s" : ""}
                </div>
              </div>
            );
          })}

          <div style={{ marginTop: "32px", paddingTop: "20px", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
            <div style={{ fontSize: "9px", letterSpacing: "0.2em", color: "#334155", marginBottom: "10px", textTransform: "uppercase" }}>Legend</div>
            {SPEAKERS.map((sp) => (
              <div key={sp} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                <div style={{ width: "12px", height: "12px", borderRadius: "2px", background: COLORS[sp].bar, boxShadow: `0 0 6px ${COLORS[sp].glow}` }} />
                <span style={{ fontSize: "10px", color: "#64748b" }}>{SPEAKER_LABELS[sp]}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Gantt Chart */}
        <div style={{ flex: 1, padding: "32px 32px 32px 0", overflowX: "auto" }}>
          {/* Time axis top */}
          <div style={{ marginLeft: "8px", marginBottom: "16px", position: "relative", height: "24px" }}>
            {ticks.map((t, i) => (
              <div key={i} style={{
                position: "absolute",
                left: `${((t - zoom.start) / viewDur) * 100}%`,
                transform: "translateX(-50%)",
                fontSize: "9px",
                color: "#475569",
                letterSpacing: "0.05em",
                whiteSpace: "nowrap",
              }}>
                {fmtTime(t)}
              </div>
            ))}
          </div>

          {/* Chart area */}
          <div
            ref={chartRef}
            style={{ marginLeft: "8px", userSelect: "none", cursor: "crosshair" }}
            onMouseDown={handleChartMouseDown}
            onMouseUp={handleChartMouseUp}
            onMouseLeave={() => { setIsDragging(false); setDragStart(null); setTooltip(null); }}
          >
            {/* Grid lines SVG overlay */}
            <div style={{ position: "relative" }}>
              <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
                {ticks.map((t, i) => (
                  <line
                    key={i}
                    className="tick-line"
                    x1={`${((t - zoom.start) / viewDur) * 100}%`}
                    x2={`${((t - zoom.start) / viewDur) * 100}%`}
                    y1="0"
                    y2="100%"
                    stroke="#94a3b8"
                    strokeWidth="1"
                  />
                ))}
              </svg>

              {SPEAKERS.map((sp, si) => {
                const col = COLORS[sp];
                const spBlocks = BLOCKS.filter(
                  (b) => b.speaker === sp && b.end > zoom.start && b.start < zoom.end
                );
                const isHovered = hoveredSpeaker === sp;
                return (
                  <div
                    key={sp}
                    className="speaker-row"
                    style={{
                      position: "relative",
                      height: "52px",
                      marginBottom: "6px",
                      borderRadius: "6px",
                      background: isHovered ? col.bg : "rgba(255,255,255,0.015)",
                      display: "flex",
                      alignItems: "center",
                    }}
                    onMouseEnter={() => setHoveredSpeaker(sp)}
                    onMouseLeave={() => setHoveredSpeaker(null)}
                  >
                    {/* Speaker label */}
                    <div style={{
                      position: "absolute",
                      left: "-132px",
                      width: "126px",
                      textAlign: "right",
                      fontSize: "11px",
                      fontWeight: 500,
                      color: isHovered ? col.glow : "#64748b",
                      letterSpacing: "0.03em",
                      transition: "color 0.2s",
                    }}>
                      {SPEAKER_LABELS[sp]}
                    </div>

                    {/* Bars */}
                    {spBlocks.map((b, bi) => {
                      const clampedStart = Math.max(b.start, zoom.start);
                      const clampedEnd = Math.min(b.end, zoom.end);
                      const leftPct = ((clampedStart - zoom.start) / viewDur) * 100;
                      const widthPct = ((clampedEnd - clampedStart) / viewDur) * 100;
                      const minWidth = widthPct < 0.2 ? 0.2 : widthPct;
                      return (
                        <div
                          key={bi}
                          className="gantt-bar"
                          style={{
                            position: "absolute",
                            left: `${leftPct}%`,
                            width: `${minWidth}%`,
                            height: "32px",
                            background: `linear-gradient(180deg, ${col.glow} 0%, ${col.bar} 100%)`,
                            boxShadow: isHovered ? `0 0 12px ${col.glow}88, 0 2px 4px rgba(0,0,0,0.4)` : `0 0 6px ${col.bar}55`,
                            opacity: hoveredSpeaker && !isHovered ? 0.2 : 0.9,
                          }}
                          onMouseMove={(e) => handleMouseMove(e, b)}
                          onMouseLeave={() => setTooltip(null)}
                        />
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Time axis bottom */}
          <div style={{ marginLeft: "8px", marginTop: "12px", position: "relative", height: "24px" }}>
            {ticks.map((t, i) => (
              <div key={i} style={{
                position: "absolute",
                left: `${((t - zoom.start) / viewDur) * 100}%`,
                transform: "translateX(-50%)",
                fontSize: "9px",
                color: "#334155",
                letterSpacing: "0.05em",
                whiteSpace: "nowrap",
              }}>
                {fmtTime(t)}
              </div>
            ))}
          </div>

          {/* Zoom indicator */}
          {(zoom.start !== 0 || zoom.end !== DURATION) && (
            <div style={{
              marginLeft: "8px",
              marginTop: "8px",
              fontSize: "10px",
              color: "#475569",
              padding: "6px 12px",
              background: "rgba(255,255,255,0.04)",
              borderRadius: "4px",
              border: "1px solid rgba(255,255,255,0.06)",
              display: "inline-block",
            }}>
              Viewing {fmtTime(zoom.start)} → {fmtTime(zoom.end)} ({fmtDuration(viewDur)})
            </div>
          )}

          {/* Mini overview map */}
          <div style={{ marginLeft: "8px", marginTop: "28px" }}>
            <div style={{ fontSize: "9px", color: "#334155", letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: "8px" }}>
              Full Session Overview
            </div>
            <div style={{ position: "relative", height: "40px", background: "rgba(255,255,255,0.03)", borderRadius: "4px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.06)" }}>
              {BLOCKS.map((b, i) => {
                const leftPct = (b.start / DURATION) * 100;
                const widthPct = Math.max(0.3, ((b.end - b.start) / DURATION) * 100);
                const col = COLORS[b.speaker];
                return (
                  <div key={i} style={{
                    position: "absolute",
                    left: `${leftPct}%`,
                    width: `${widthPct}%`,
                    height: "100%",
                    background: col.bar,
                    opacity: 0.7,
                    top: 0,
                  }} />
                );
              })}
              {/* Zoom window indicator */}
              <div style={{
                position: "absolute",
                left: `${(zoom.start / DURATION) * 100}%`,
                width: `${((zoom.end - zoom.start) / DURATION) * 100}%`,
                height: "100%",
                background: "rgba(255,255,255,0.12)",
                border: "1px solid rgba(255,255,255,0.3)",
                borderRadius: "2px",
                top: 0,
                pointerEvents: "none",
              }} />
            </div>
          </div>
        </div>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div style={{
          position: "fixed",
          left: tooltip.x + 14,
          top: tooltip.y - 10,
          background: "#0f172a",
          border: `1px solid ${COLORS[tooltip.block.speaker].bar}44`,
          borderRadius: "8px",
          padding: "10px 14px",
          fontSize: "11px",
          lineHeight: "1.7",
          pointerEvents: "none",
          zIndex: 9999,
          boxShadow: `0 4px 24px rgba(0,0,0,0.6), 0 0 12px ${COLORS[tooltip.block.speaker].bar}22`,
          minWidth: "160px",
        }}>
          <div style={{ fontWeight: 600, color: COLORS[tooltip.block.speaker].glow, marginBottom: "4px" }}>
            {SPEAKER_LABELS[tooltip.block.speaker]}
          </div>
          <div style={{ color: "#64748b" }}>Start: <span style={{ color: "#94a3b8" }}>{fmtTime(tooltip.block.start)}</span></div>
          <div style={{ color: "#64748b" }}>End: <span style={{ color: "#94a3b8" }}>{fmtTime(tooltip.block.end)}</span></div>
          <div style={{ color: "#64748b" }}>Duration: <span style={{ color: "#94a3b8" }}>{fmtDuration(tooltip.block.end - tooltip.block.start)}</span></div>
        </div>
      )}
    </div>
  );
}
