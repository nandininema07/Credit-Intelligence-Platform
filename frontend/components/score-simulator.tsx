"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Calculator, RotateCcw, TrendingUp, TrendingDown } from "lucide-react"

interface ScoreSimulatorProps {
  company: string
  currentScore: number
}

const factors = [
  { name: "Revenue Growth", current: 15, min: -20, max: 50, unit: "%", weight: 0.25 },
  { name: "Debt-to-Equity", current: 0.3, min: 0, max: 2, unit: "x", weight: -0.3 },
  { name: "News Sentiment", current: -0.2, min: -1, max: 1, unit: "", weight: 0.15 },
  { name: "Market Volatility", current: 0.25, min: 0, max: 1, unit: "", weight: -0.2 },
  { name: "Cash Flow", current: 8.5, min: -5, max: 20, unit: "B", weight: 0.2 },
]

export function ScoreSimulator({ company, currentScore }: ScoreSimulatorProps) {
  const [adjustedFactors, setAdjustedFactors] = useState(
    factors.reduce((acc, factor) => ({ ...acc, [factor.name]: factor.current }), {}),
  )
  const [simulatedScore, setSimulatedScore] = useState(currentScore)
  const [scenarioName, setScenarioName] = useState("")
  const [savedScenarios, setSavedScenarios] = useState<Array<{ name: string; score: number; factors: any }>>([])

  useEffect(() => {
    // Calculate new score based on factor changes
    let scoreChange = 0
    factors.forEach((factor) => {
      const currentValue = adjustedFactors[factor.name] || factor.current
      const change = (currentValue - factor.current) / (factor.max - factor.min)
      scoreChange += change * factor.weight * 20 // Scale the impact
    })

    const newScore = Math.max(0, Math.min(100, currentScore + scoreChange))
    setSimulatedScore(Math.round(newScore))
  }, [adjustedFactors, currentScore])

  const resetFactors = () => {
    setAdjustedFactors(factors.reduce((acc, factor) => ({ ...acc, [factor.name]: factor.current }), {}))
  }

  const saveScenario = () => {
    if (scenarioName.trim()) {
      setSavedScenarios((prev) => [
        ...prev,
        {
          name: scenarioName,
          score: simulatedScore,
          factors: { ...adjustedFactors },
        },
      ])
      setScenarioName("")
    }
  }

  const loadScenario = (scenario: any) => {
    setAdjustedFactors(scenario.factors)
  }

  const scoreChange = simulatedScore - currentScore
  const isPositive = scoreChange > 0

  return (
    <div className="space-y-6">
      {/* Score Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold text-muted-foreground">Current Score</div>
            <div className="text-4xl font-playfair font-bold mt-2">{currentScore}</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold text-foreground">Simulated Score</div>
            <div className="text-4xl font-playfair font-bold mt-2">{simulatedScore}</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold">Impact</div>
            <div className="flex items-center justify-center gap-2 mt-2">
              <Badge variant={isPositive ? "default" : "destructive"} className="text-lg px-3 py-1">
                {isPositive ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
                {isPositive ? "+" : ""}
                {scoreChange}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Factor Adjustments */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Calculator className="w-5 h-5 text-primary" />
              Assumption Analysis
            </CardTitle>
            <Button variant="outline" onClick={resetFactors}>
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {factors.map((factor) => (
            <div key={factor.name} className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="font-medium">{factor.name}</Label>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">
                    {adjustedFactors[factor.name]?.toFixed(factor.unit === "%" ? 0 : 1)}
                    {factor.unit}
                  </span>
                  <Badge variant="outline" className="text-xs">
                    Weight: {Math.abs(factor.weight * 100).toFixed(0)}%
                  </Badge>
                </div>
              </div>
              <Slider
                value={[adjustedFactors[factor.name] || factor.current]}
                onValueChange={(value) => setAdjustedFactors((prev) => ({ ...prev, [factor.name]: value[0] }))}
                min={factor.min}
                max={factor.max}
                step={factor.unit === "%" ? 1 : 0.1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>
                  {factor.min}
                  {factor.unit}
                </span>
                <span>
                  Current: {factor.current}
                  {factor.unit}
                </span>
                <span>
                  {factor.max}
                  {factor.unit}
                </span>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Scenario Management */}
      <Card>
        <CardHeader>
          <CardTitle>Scenario Management</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Scenario name (e.g., 'Q4 Growth Scenario')"
              value={scenarioName}
              onChange={(e) => setScenarioName(e.target.value)}
            />
            <Button onClick={saveScenario} disabled={!scenarioName.trim()}>
              Save Scenario
            </Button>
          </div>

          {savedScenarios.length > 0 && (
            <div className="space-y-2">
              <Label>Saved Scenarios</Label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {savedScenarios.map((scenario, index) => (
                  <Card key={index} className="cursor-pointer hover:bg-accent" onClick={() => loadScenario(scenario)}>
                    <CardContent className="p-3">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{scenario.name}</span>
                        <Badge variant="outline">Score: {scenario.score}</Badge>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
