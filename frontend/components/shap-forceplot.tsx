import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { TrendingUp, TrendingDown } from "lucide-react"

const shapData = [
  { factor: "Strong Earnings Report", impact: 8.5, type: "positive" },
  { factor: "Revenue Growth", impact: 5.2, type: "positive" },
  { factor: "Market Sentiment", impact: 3.1, type: "positive" },
  { factor: "Supply Chain Issues", impact: -2.8, type: "negative" },
  { factor: "Regulatory Concerns", impact: -1.5, type: "negative" },
]

export function ShapForceplot() {
  const baseScore = 75
  const totalImpact = shapData.reduce((sum, item) => sum + item.impact, 0)
  const finalScore = baseScore + totalImpact

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
        <span className="text-sm font-medium">Base Score</span>
        <span className="font-bold">{baseScore}</span>
      </div>

      <div className="space-y-3">
        {shapData.map((item, index) => (
          <div key={index} className="flex items-center gap-3">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                {item.type === "positive" ? (
                  <TrendingUp className="w-4 h-4 text-primary" />
                ) : (
                  <TrendingDown className="w-4 h-4 text-destructive" />
                )}
                <span className="text-sm font-medium">{item.factor}</span>
              </div>
              <div className="flex items-center gap-2">
                <Progress value={Math.abs(item.impact) * 10} className="flex-1 h-2" />
                <Badge variant={item.type === "positive" ? "default" : "destructive"} className="text-xs">
                  {item.impact > 0 ? "+" : ""}
                  {item.impact}
                </Badge>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between p-4 bg-primary/10 rounded-lg border border-primary/20">
        <span className="text-sm font-medium">Final Score</span>
        <span className="font-bold text-lg text-primary">{finalScore.toFixed(1)}</span>
      </div>
    </div>
  )
}
