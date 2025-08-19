"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { TrendingUp, TrendingDown, Info, FileText, MessageSquare, Globe, BarChart3 } from "lucide-react"
import { CompanyScoreChart } from "@/components/company-score-chart"
import { ShapForceplot } from "@/components/shap-forceplot"
import { KeyFactorsTable } from "@/components/key-factors-table"
import { RecentEventsFeed } from "@/components/recent-events-feed"
import { GlobalDataInsights } from "@/components/global-data-insights"
import { ScoreSimulator } from "@/components/score-simulator"
import { PeerComparison } from "@/components/peer-comparison"

interface CompanyDeepDiveProps {
  selectedCompany: string | null
  onCompanyChange: (company: string) => void
}

const companies = [
  "Apple Inc.",
  "Microsoft Corp.",
  "Tesla Inc.",
  "Amazon.com Inc.",
  "Google (Alphabet)",
  "Meta Platforms",
  "Netflix Inc.",
  "JPMorgan Chase",
]

export function CompanyDeepDive({ selectedCompany, onCompanyChange }: CompanyDeepDiveProps) {
  const [timeframe, setTimeframe] = useState("1M")
  const [activeTab, setActiveTab] = useState("analysis")

  const currentCompany = selectedCompany || "Apple Inc."
  const currentScore = 89
  const dailyChange = "+2.5%"
  const isPositive = dailyChange.startsWith("+")

  return (
    <div className="p-6 space-y-6 overflow-auto h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-playfair text-3xl font-bold text-foreground">Company Deep Dive</h1>
          <p className="text-muted-foreground">Detailed credit risk analysis with explainable AI insights</p>
        </div>
        <div className="flex items-center gap-4">
          <Select value={currentCompany} onValueChange={onCompanyChange}>
            <SelectTrigger className="w-64">
              <SelectValue placeholder="Select a company" />
            </SelectTrigger>
            <SelectContent>
              {companies.map((company) => (
                <SelectItem key={company} value={company}>
                  {company}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex gap-2 border-b">
        <Button
          variant={activeTab === "analysis" ? "default" : "ghost"}
          onClick={() => setActiveTab("analysis")}
          className="rounded-b-none"
        >
          <Info className="w-4 h-4 mr-2" />
          Analysis
        </Button>
        <Button
          variant={activeTab === "simulator" ? "default" : "ghost"}
          onClick={() => setActiveTab("simulator")}
          className="rounded-b-none"
        >
          <BarChart3 className="w-4 h-4 mr-2" />
          Score Simulator
        </Button>
        <Button
          variant={activeTab === "peers" ? "default" : "ghost"}
          onClick={() => setActiveTab("peers")}
          className="rounded-b-none"
        >
          <Globe className="w-4 h-4 mr-2" />
          Peer Comparison
        </Button>
      </div>

      {/* Company Header */}
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-playfair text-2xl font-bold">{currentCompany}</h2>
              <p className="text-muted-foreground">Real-time Credit Risk Score</p>
            </div>
            <div className="text-right">
              <div className="flex items-center gap-2">
                <span className="font-playfair text-4xl font-bold">{currentScore}</span>
                <Badge variant={isPositive ? "default" : "destructive"} className="text-sm">
                  {isPositive ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
                  {dailyChange}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">Daily Change</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {activeTab === "analysis" && (
        <>
          {/* Main Score Chart */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Historical Credit Score</CardTitle>
                <div className="flex gap-2">
                  {["1D", "1W", "1M", "3M", "1Y"].map((period) => (
                    <Button
                      key={period}
                      variant={timeframe === period ? "default" : "outline"}
                      size="sm"
                      onClick={() => setTimeframe(period)}
                    >
                      {period}
                    </Button>
                  ))}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <CompanyScoreChart timeframe={timeframe} />
            </CardContent>
          </Card>

          {/* Explainable AI Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Info className="w-5 h-5 text-primary" />
                  SHAP Force Plot Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ShapForceplot />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-primary" />
                  Key Factors Breakdown
                </CardTitle>
              </CardHeader>
              <CardContent>
                <KeyFactorsTable />
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageSquare className="w-5 h-5 text-primary" />
                  Recent Events Feed
                </CardTitle>
              </CardHeader>
              <CardContent>
                <RecentEventsFeed company={currentCompany} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Globe className="w-5 h-5 text-primary" />
                  Global Data Insights
                </CardTitle>
              </CardHeader>
              <CardContent>
                <GlobalDataInsights company={currentCompany} />
              </CardContent>
            </Card>
          </div>
        </>
      )}

      {activeTab === "simulator" && <ScoreSimulator company={currentCompany} currentScore={currentScore} />}

      {activeTab === "peers" && <PeerComparison company={currentCompany} />}
    </div>
  )
}
