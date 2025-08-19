"use client"

import { LineChart, Line, ResponsiveContainer } from "recharts"

interface MiniChartProps {
  data: number[]
  trend: "up" | "down" | "stable"
}

export function MiniChart({ data, trend }: MiniChartProps) {
  const chartData = data.map((value, index) => ({ value, index }))

  const getColor = () => {
    switch (trend) {
      case "up":
        return "hsl(var(--primary))"
      case "down":
        return "hsl(var(--destructive))"
      default:
        return "hsl(var(--muted-foreground))"
    }
  }

  return (
    <div className="h-8 w-20">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <Line type="monotone" dataKey="value" stroke={getColor()} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
