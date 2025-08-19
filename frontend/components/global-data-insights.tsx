"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Globe, Languages, TrendingUp, TrendingDown, MapPin } from "lucide-react"

interface GlobalDataInsightsProps {
  company: string
}

const dataPoints = [
  {
    id: 1,
    location: "São Paulo, Brazil",
    source: "Reuters - Brazil",
    originalLanguage: "Portuguese",
    sentiment: "Negative",
    sentimentScore: -0.7,
    snippet: "Apple enfrenta desafios regulatórios no mercado brasileiro",
    translation: "Apple faces regulatory challenges in Brazilian market",
    timestamp: "2 hours ago",
    coordinates: { x: 35, y: 65 },
  },
  {
    id: 2,
    location: "Tokyo, Japan",
    source: "Nikkei Asia",
    originalLanguage: "Japanese",
    sentiment: "Positive",
    sentimentScore: 0.6,
    snippet: "アップルの新製品が日本市場で好調",
    translation: "Apple's new products performing well in Japanese market",
    timestamp: "4 hours ago",
    coordinates: { x: 85, y: 35 },
  },
  {
    id: 3,
    location: "London, UK",
    source: "Financial Times",
    originalLanguage: "English",
    sentiment: "Neutral",
    sentimentScore: 0.1,
    snippet: "Apple maintains steady growth in European markets",
    translation: "Apple maintains steady growth in European markets",
    timestamp: "6 hours ago",
    coordinates: { x: 50, y: 25 },
  },
]

export function GlobalDataInsights({ company }: GlobalDataInsightsProps) {
  const [selectedPoint, setSelectedPoint] = useState<number | null>(null)
  const [showOriginal, setShowOriginal] = useState<{ [key: number]: boolean }>({})

  const toggleOriginal = (id: number) => {
    setShowOriginal((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  return (
    <div className="space-y-4">
      {/* Interactive World Map */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Globe className="w-5 h-5 text-primary" />
            Real-time Data Sources Map
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative bg-slate-50 rounded-lg p-4 h-64 overflow-hidden">
            {/* Simplified world map background */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-green-50 rounded-lg"></div>

            {/* Data points on map */}
            {dataPoints.map((point) => (
              <div
                key={point.id}
                className={`absolute w-4 h-4 rounded-full cursor-pointer transition-all duration-300 ${
                  point.sentiment === "Positive"
                    ? "bg-green-500 animate-pulse"
                    : point.sentiment === "Negative"
                      ? "bg-red-500 animate-pulse"
                      : "bg-yellow-500 animate-pulse"
                }`}
                style={{
                  left: `${point.coordinates.x}%`,
                  top: `${point.coordinates.y}%`,
                  transform: "translate(-50%, -50%)",
                }}
                onClick={() => setSelectedPoint(selectedPoint === point.id ? null : point.id)}
              >
                {selectedPoint === point.id && (
                  <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 bg-white p-3 rounded-lg shadow-lg border z-10 w-64">
                    <div className="text-sm">
                      <div className="font-semibold">{point.location}</div>
                      <div className="text-muted-foreground">{point.source}</div>
                      <Badge
                        variant={
                          point.sentiment === "Positive"
                            ? "default"
                            : point.sentiment === "Negative"
                              ? "destructive"
                              : "secondary"
                        }
                        className="mt-1"
                      >
                        {point.sentiment} ({point.sentimentScore > 0 ? "+" : ""}
                        {point.sentimentScore})
                      </Badge>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Map legend */}
            <div className="absolute bottom-2 right-2 bg-white p-2 rounded shadow text-xs">
              <div className="flex items-center gap-1 mb-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Positive</span>
              </div>
              <div className="flex items-center gap-1 mb-1">
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                <span>Negative</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                <span>Neutral</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Data Sources List */}
      <div className="space-y-3">
        {dataPoints.map((point) => (
          <Card key={point.id} className="border-l-4 border-l-primary">
            <CardContent className="p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <MapPin className="w-4 h-4 text-muted-foreground" />
                    <span className="font-medium">{point.location}</span>
                    <Badge variant="outline" className="text-xs">
                      <Languages className="w-3 h-3 mr-1" />
                      {point.originalLanguage}
                    </Badge>
                    <Badge
                      variant={
                        point.sentiment === "Positive"
                          ? "default"
                          : point.sentiment === "Negative"
                            ? "destructive"
                            : "secondary"
                      }
                    >
                      {point.sentiment === "Positive" ? (
                        <TrendingUp className="w-3 h-3 mr-1" />
                      ) : point.sentiment === "Negative" ? (
                        <TrendingDown className="w-3 h-3 mr-1" />
                      ) : null}
                      {point.sentiment}
                    </Badge>
                  </div>

                  <div className="text-sm text-muted-foreground mb-2">
                    {point.source} • {point.timestamp}
                  </div>

                  <div className="text-sm">
                    {showOriginal[point.id] ? (
                      <div className="space-y-2">
                        <div className="font-medium text-muted-foreground">Original ({point.originalLanguage}):</div>
                        <div className="italic bg-slate-50 p-2 rounded">{point.snippet}</div>
                        <div className="font-medium text-muted-foreground">Translation:</div>
                        <div>{point.translation}</div>
                      </div>
                    ) : (
                      <div>{point.translation}</div>
                    )}
                  </div>
                </div>

                <Button variant="ghost" size="sm" onClick={() => toggleOriginal(point.id)} className="ml-4">
                  <Languages className="w-4 h-4 mr-1" />
                  {showOriginal[point.id] ? "Hide" : "Show"} Original
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
