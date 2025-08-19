"use client"

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts"

interface CompanyScoreChartProps {
  timeframe: string
}

const generateData = (timeframe: string) => {
  const baseData = {
    "1D": [
      { time: "09:00", score: 87, event: null },
      { time: "10:00", score: 86, event: null },
      { time: "11:00", score: 88, event: "Earnings Report" },
      { time: "12:00", score: 89, event: null },
      { time: "13:00", score: 89, event: null },
      { time: "14:00", score: 90, event: null },
      { time: "15:00", score: 89, event: null },
      { time: "16:00", score: 89, event: null },
    ],
    "1W": [
      { time: "Mon", score: 85, event: null },
      { time: "Tue", score: 87, event: null },
      { time: "Wed", score: 88, event: "Earnings Report" },
      { time: "Thu", score: 89, event: null },
      { time: "Fri", score: 89, event: null },
    ],
    "1M": [
      { time: "Week 1", score: 82, event: null },
      { time: "Week 2", score: 85, event: null },
      { time: "Week 3", score: 87, event: "Product Launch" },
      { time: "Week 4", score: 89, event: "Earnings Report" },
    ],
  }

  return baseData[timeframe as keyof typeof baseData] || baseData["1M"]
}

export function CompanyScoreChart({ timeframe }: CompanyScoreChartProps) {
  const data = generateData(timeframe)

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
          <XAxis dataKey="time" className="text-xs" tick={{ fill: "hsl(var(--muted-foreground))" }} />
          <YAxis
            domain={["dataMin - 5", "dataMax + 5"]}
            className="text-xs"
            tick={{ fill: "hsl(var(--muted-foreground))" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
            }}
            formatter={(value, name) => [value, "Credit Score"]}
          />
          <Line
            type="monotone"
            dataKey="score"
            stroke="hsl(var(--primary))"
            strokeWidth={3}
            dot={(props) => {
              const { cx, cy, payload } = props
              return payload.event ? (
                <circle cx={cx} cy={cy} r={6} fill="hsl(var(--accent))" stroke="hsl(var(--primary))" strokeWidth={2} />
              ) : (
                <circle cx={cx} cy={cy} r={4} fill="hsl(var(--primary))" strokeWidth={2} />
              )
            }}
          />
          {data.map((point, index) =>
            point.event ? (
              <ReferenceLine
                key={index}
                x={point.time}
                stroke="hsl(var(--accent))"
                strokeDasharray="2 2"
                label={{ value: point.event, position: "top" }}
              />
            ) : null,
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
