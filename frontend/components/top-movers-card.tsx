"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { TrendingUp, TrendingDown, Eye } from "lucide-react"

interface TopMoversCardProps {
  title: string
  type: "positive" | "negative"
  onCompanyClick: (companyName: string) => void
}

const mockData = {
  positive: [
    { name: "Apple Inc.", score: 89, change: "+12.5%", reason: "Strong earnings report" },
    { name: "Microsoft Corp.", score: 87, change: "+8.3%", reason: "Cloud growth momentum" },
    { name: "Tesla Inc.", score: 82, change: "+15.2%", reason: "Production milestone" },
    { name: "Amazon.com Inc.", score: 85, change: "+6.7%", reason: "AWS expansion" },
  ],
  negative: [
    { name: "Meta Platforms", score: 68, change: "-9.4%", reason: "Regulatory concerns" },
    { name: "Netflix Inc.", score: 71, change: "-7.8%", reason: "Subscriber decline" },
    { name: "PayPal Holdings", score: 73, change: "-11.2%", reason: "Competition pressure" },
    { name: "Zoom Video", score: 69, change: "-8.9%", reason: "Post-pandemic adjustment" },
  ],
}

export function TopMoversCard({ title, type, onCompanyClick }: TopMoversCardProps) {
  const data = mockData[type]
  const Icon = type === "positive" ? TrendingUp : TrendingDown
  const colorClass = type === "positive" ? "text-primary" : "text-destructive"

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Icon className={`w-5 h-5 ${colorClass}`} />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {data.map((company, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-sm">{company.name}</span>
                  <Badge variant={type === "positive" ? "default" : "destructive"} className="text-xs">
                    {company.change}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground mt-1">{company.reason}</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg">{company.score}</span>
                <Button size="sm" variant="ghost" onClick={() => onCompanyClick(company.name)}>
                  <Eye className="w-4 h-4" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
