import { Badge } from "@/components/ui/badge"
import { TrendingUp, TrendingDown, Minus } from "lucide-react"

const factors = [
  {
    factor: "Financial Performance",
    impact: "High Positive",
    value: "+8.5",
    description: "Strong Q4 earnings beat expectations",
  },
  {
    factor: "Market Position",
    impact: "Medium Positive",
    value: "+5.2",
    description: "Increased market share in key segments",
  },
  {
    factor: "Debt Levels",
    impact: "Low Negative",
    value: "-1.8",
    description: "Slight increase in debt-to-equity ratio",
  },
  {
    factor: "Industry Outlook",
    impact: "Medium Positive",
    value: "+3.1",
    description: "Favorable industry growth projections",
  },
  {
    factor: "ESG Score",
    impact: "Neutral",
    value: "0.0",
    description: "Stable environmental and governance metrics",
  },
]

export function KeyFactorsTable() {
  return (
    <div className="space-y-3">
      {factors.map((factor, index) => (
        <div key={index} className="p-3 rounded-lg border border-border hover:bg-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center gap-2">
              {factor.impact.includes("Positive") ? (
                <TrendingUp className="w-4 h-4 text-primary" />
              ) : factor.impact.includes("Negative") ? (
                <TrendingDown className="w-4 h-4 text-destructive" />
              ) : (
                <Minus className="w-4 h-4 text-muted-foreground" />
              )}
              <span className="font-medium text-sm">{factor.factor}</span>
            </div>
            <Badge
              variant={
                factor.impact.includes("Positive")
                  ? "default"
                  : factor.impact.includes("Negative")
                    ? "destructive"
                    : "secondary"
              }
              className="text-xs"
            >
              {factor.value}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground pl-6">{factor.description}</p>
        </div>
      ))}
    </div>
  )
}
