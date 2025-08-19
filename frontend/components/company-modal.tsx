"use client"

import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { TrendingUp, TrendingDown, X } from "lucide-react"
import { CompanyScoreChart } from "@/components/company-score-chart"
import { ShapForceplot } from "@/components/shap-forceplot"

interface CompanyModalProps {
  companyName: string
  onClose: () => void
}

export function CompanyModal({ companyName, onClose }: CompanyModalProps) {
  const mockData = {
    score: 89,
    change: "+2.5%",
    isPositive: true,
  }

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <DialogTitle className="font-playfair text-2xl">{companyName} - Deep Dive Analysis</DialogTitle>
            <Button variant="ghost" size="sm" onClick={onClose}>
              <X className="w-4 h-4" />
            </Button>
          </div>
        </DialogHeader>

        <div className="space-y-6">
          {/* Score Header */}
          <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
            <div>
              <h3 className="font-medium text-lg">Current Credit Score</h3>
              <p className="text-sm text-muted-foreground">Real-time assessment</p>
            </div>
            <div className="text-right">
              <div className="flex items-center gap-2">
                <span className="font-playfair text-3xl font-bold">{mockData.score}</span>
                <Badge variant={mockData.isPositive ? "default" : "destructive"}>
                  {mockData.isPositive ? (
                    <TrendingUp className="w-4 h-4 mr-1" />
                  ) : (
                    <TrendingDown className="w-4 h-4 mr-1" />
                  )}
                  {mockData.change}
                </Badge>
              </div>
            </div>
          </div>

          {/* Chart */}
          <div>
            <h3 className="font-medium mb-4">7-Day Score Trend</h3>
            <div className="h-64 border rounded-lg p-4">
              <CompanyScoreChart timeframe="1W" />
            </div>
          </div>

          {/* SHAP Analysis */}
          <div>
            <h3 className="font-medium mb-4">AI Explainability - Key Drivers</h3>
            <div className="border rounded-lg p-4">
              <ShapForceplot />
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex gap-2 pt-4 border-t">
            <Button>Add to Watchlist</Button>
            <Button variant="outline">Set Alert</Button>
            <Button variant="outline">Generate Report</Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
