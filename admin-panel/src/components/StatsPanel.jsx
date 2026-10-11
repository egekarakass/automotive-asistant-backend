import { useState, useEffect } from "react";
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Legend
} from "recharts";
import { useLang } from "../i18n";

const COLORS = ["#6366f1", "#10b981", "#f59e0b", "#64748b"];

function StatsPanel({ apiBase, refreshKey, token, onAuthError }) {
  const { t } = useLang();
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetch(`${apiBase}/admin/stats`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (res.status === 401) {
          onAuthError();
          return null;
        }
        return res.json();
      })
      .then((data) => data && setStats(data))
      .catch(() => setStats(null));
  }, [apiBase, token, refreshKey]);

  if (!stats) return null;

  const evalSummary = stats.evaluation_summary || {};
  const toolTotal = evalSummary.tool_total || 0;
  const answerTotal = evalSummary.total || 0;
  const correctToolPct = toolTotal ? Math.round((evalSummary.correct_tool / toolTotal) * 100) : 0;
  const correctAnswerPct = answerTotal ? Math.round((evalSummary.correct_answer / answerTotal) * 100) : 0;
  const langData = stats.language_distribution || [];

  return (
    <div className="stats-panel">
      <div className="stat-card">
        <h4>{t("statToolRate")}</h4>
        <div className="stat-number">{correctToolPct}%</div>
        <span className="stat-sub">{toolTotal} {t("statToolSub")}</span>
      </div>
      <div className="stat-card">
        <h4>{t("statAnswerRate")}</h4>
        <div className="stat-number">{correctAnswerPct}%</div>
        <span className="stat-sub">{answerTotal} {t("statAnswerSub")}</span>
      </div>

      <div className="chart-card">
        <h4>{t("modelUsage")}</h4>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={stats.model_usage}>
            <XAxis dataKey="model_used" tick={{ fontSize: 10 }} hide />
            <YAxis allowDecimals={false} />
            <Tooltip
                cursor={{ fill: 'var(--border-color)', opacity: 0.6 }}
                contentStyle={{
                backgroundColor: 'var(--bg-card)',
                borderColor: 'var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.3)'
              }}
              itemStyle={{ color: 'var(--text-primary)' }}
            />
            <Bar dataKey="c" fill="#4f46e5" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-card">
        <h4>{t("langDistribution")}</h4>
        <ResponsiveContainer width="100%" height={180}>
          <PieChart>
            <Pie
              data={langData}
              cx="50%"
              cy="50%"
              outerRadius={70}
              paddingAngle={2}
              dataKey="c"
              nameKey="language"
              stroke="var(--bg-card)"
              labelLine={false}
              label={false}
            >
              {langData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--bg-card)",
                borderColor: "var(--border-color)",
                borderRadius: "8px",
                color: "var(--text-primary)",
                boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.3)",
              }}
              itemStyle={{ color: "var(--text-primary)" }}
            />
            <Legend
              iconType="circle"
              wrapperStyle={{ fontSize: "12px", color: "var(--text-secondary)" }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default StatsPanel;