"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Star, Eye, Plus } from "lucide-react"
import { MiniChart } from "@/components/mini-chart"

interface WatchlistCardProps {
  onCompanyClick: (companyName: string) => void
}

const watchlistData = [
  { name: "Google (Alphabet)", score: 84, trend: "up", data: [78, 80, 82, 81, 84] },
  { name: "JPMorgan Chase", score: 79, trend: "stable", data: [77, 78, 79, 78, 79] },
  { name: "Johnson & Johnson", score: 88, trend: "up", data: [85, 86, 87, 86, 88] },
  { name: "Berkshire Hathaway", score: 91, trend: "down", data: [93, 92, 91, 92, 91] },
]

export function WatchlistCard({ onCompanyClick }: WatchlistCardProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Star className="w-5 h-5 text-primary" />
            Watchlist
          </CardTitle>
          <Button size="sm" variant="outline">
            <Plus className="w-4 h-4 mr-1" />
            Add Company
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {watchlistData.map((company, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-sm">{company.name}</span>
                  <Badge
                    variant={
                      company.trend === "up" ? "default" : company.trend === "down" ? "destructive" : "secondary"
                    }
                    className="text-xs"
                  >
                    {company.trend}
                  </Badge>
                </div>
                <div className="mt-2">
                  <MiniChart data={company.data} trend={company.trend} />
                </div>
              </div>
              <div className="flex items-center gap-2 ml-4">
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
