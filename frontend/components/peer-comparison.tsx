"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from "recharts"
import { TrendingUp, TrendingDown, Users, BarChart3 } from "lucide-react"

interface PeerComparisonProps {
  company: string
}

const peerGroups = {
  "Apple Inc.": ["Microsoft Corp.", "Google (Alphabet)", "Meta Platforms", "Amazon.com Inc."],
  "Tesla Inc.": ["Ford Motor Co.", "General Motors", "Rivian", "Lucid Motors"],
  "JPMorgan Chase": ["Bank of America", "Wells Fargo", "Citigroup", "Goldman Sachs"],
}

const historicalData = [
  {
    month: "Jan",
    "Apple Inc.": 85,
    "Microsoft Corp.": 88,
    "Google (Alphabet)": 82,
    "Meta Platforms": 79,
    "Industry Avg": 83,
  },
  {
    month: "Feb",
    "Apple Inc.": 87,
    "Microsoft Corp.": 89,
    "Google (Alphabet)": 84,
    "Meta Platforms": 81,
    "Industry Avg": 85,
  },
  {
    month: "Mar",
    "Apple Inc.": 89,
    "Microsoft Corp.": 91,
    "Google (Alphabet)": 86,
    "Meta Platforms": 83,
    "Industry Avg": 87,
  },
  {
    month: "Apr",
    "Apple Inc.": 88,
    "Microsoft Corp.": 90,
    "Google (Alphabet)": 85,
    "Meta Platforms": 82,
    "Industry Avg": 86,
  },
  {
    month: "May",
    "Apple Inc.": 90,
    "Microsoft Corp.": 92,
    "Google (Alphabet)": 87,
    "Meta Platforms": 84,
    "Industry Avg": 88,
  },
  {
    month: "Jun",
    "Apple Inc.": 89,
    "Microsoft Corp.": 91,
    "Google (Alphabet)": 86,
    "Meta Platforms": 85,
    "Industry Avg": 88,
  },
]

const radarData = [
  {
    factor: "Financial Leverage",
    "Apple Inc.": 85,
    "Microsoft Corp.": 90,
    "Google (Alphabet)": 88,
    "Industry Avg": 82,
  },
  { factor: "Market Sentiment", "Apple Inc.": 78, "Microsoft Corp.": 85, "Google (Alphabet)": 80, "Industry Avg": 75 },
  { factor: "Operational Risk", "Apple Inc.": 92, "Microsoft Corp.": 88, "Google (Alphabet)": 85, "Industry Avg": 80 },
  { factor: "Liquidity", "Apple Inc.": 88, "Microsoft Corp.": 85, "Google (Alphabet)": 90, "Industry Avg": 78 },
  { factor: "Growth Potential", "Apple Inc.": 82, "Microsoft Corp.": 88, "Google (Alphabet)": 85, "Industry Avg": 75 },
  { factor: "Regulatory Risk", "Apple Inc.": 70, "Microsoft Corp.": 75, "Google (Alphabet)": 65, "Industry Avg": 72 },
]

export function PeerComparison({ company }: PeerComparisonProps) {
  const [selectedPeers, setSelectedPeers] = useState<string[]>(
    peerGroups[company as keyof typeof peerGroups]?.slice(0, 3) || [],
  )
  const [viewType, setViewType] = useState<"trends" | "factors">("trends")

  const availablePeers = peerGroups[company as keyof typeof peerGroups] || []
  const currentScore = 89
  const industryAvg = 88

  const peerScores = [
    { name: "Microsoft Corp.", score: 91, change: "+1.2%" },
    { name: "Google (Alphabet)", score: 86, change: "-0.8%" },
    { name: "Meta Platforms", score: 85, change: "+2.1%" },
  ]

  return (
    <div className="space-y-6">
      {/* Header with Peer Selection */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Users className="w-5 h-5 text-primary" />
              Peer & Industry Comparison
            </CardTitle>
            <div className="flex gap-2">
              <Button
                variant={viewType === "trends" ? "default" : "outline"}
                onClick={() => setViewType("trends")}
                size="sm"
              >
                <TrendingUp className="w-4 h-4 mr-2" />
                Trends
              </Button>
              <Button
                variant={viewType === "factors" ? "default" : "outline"}
                onClick={() => setViewType("factors")}
                size="sm"
              >
                <BarChart3 className="w-4 h-4 mr-2" />
                Risk Factors
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium">Compare with:</span>
            <div className="flex gap-2 flex-wrap">
              {availablePeers.map((peer) => (
                <Badge
                  key={peer}
                  variant={selectedPeers.includes(peer) ? "default" : "outline"}
                  className="cursor-pointer"
                  onClick={() => {
                    setSelectedPeers(
                      (prev) => (prev.includes(peer) ? prev.filter((p) => p !== peer) : [...prev, peer].slice(0, 4)), // Max 4 peers
                    )
                  }}
                >
                  {peer}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Score Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border-2 border-primary">
          <CardContent className="p-4 text-center">
            <div className="text-sm font-medium text-primary">{company}</div>
            <div className="text-2xl font-bold mt-1">{currentScore}</div>
            <Badge variant="default" className="mt-2">
              <TrendingUp className="w-3 h-3 mr-1" />
              +2.5%
            </Badge>
          </CardContent>
        </Card>

        {peerScores.map((peer) => (
          <Card key={peer.name}>
            <CardContent className="p-4 text-center">
              <div className="text-sm font-medium text-muted-foreground">{peer.name}</div>
              <div className="text-2xl font-bold mt-1">{peer.score}</div>
              <Badge variant={peer.change.startsWith("+") ? "default" : "destructive"} className="mt-2">
                {peer.change.startsWith("+") ? (
                  <TrendingUp className="w-3 h-3 mr-1" />
                ) : (
                  <TrendingDown className="w-3 h-3 mr-1" />
                )}
                {peer.change}
              </Badge>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Industry Average */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm text-muted-foreground">Industry Average</span>
              <div className="text-xl font-bold">{industryAvg}</div>
            </div>
            <Badge variant="secondary">Technology Sector</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Comparison Charts */}
      {viewType === "trends" ? (
        <Card>
          <CardHeader>
            <CardTitle>Credit Score Trends Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historicalData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis domain={[70, 95]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey={company} stroke="#10b981" strokeWidth={3} />
                  {selectedPeers.map((peer, index) => (
                    <Line
                      key={peer}
                      type="monotone"
                      dataKey={peer}
                      stroke={`hsl(${index * 60 + 200}, 70%, 50%)`}
                      strokeWidth={2}
                    />
                  ))}
                  <Line type="monotone" dataKey="Industry Avg" stroke="#6b7280" strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Risk Factors Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="factor" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  <Radar name={company} dataKey={company} stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                  <Radar
                    name="Microsoft Corp."
                    dataKey="Microsoft Corp."
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    fillOpacity={0.2}
                  />
                  <Radar
                    name="Google (Alphabet)"
                    dataKey="Google (Alphabet)"
                    stroke="#f59e0b"
                    fill="#f59e0b"
                    fillOpacity={0.2}
                  />
                  <Radar name="Industry Avg" dataKey="Industry Avg" stroke="#6b7280" strokeDasharray="3 3" />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
