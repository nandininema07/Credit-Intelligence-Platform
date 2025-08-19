"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AlertTriangle, TrendingUp, TrendingDown, Clock, Share, Plus, VolumeX, MoreHorizontal } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

interface AlertsFeedProps {
  onCompanyClick: (companyName: string) => void
}

const alerts = [
  {
    id: 1,
    company: "Tesla Inc.",
    type: "positive",
    message: "Score increased 15% due to production milestone achievement",
    time: "2 min ago",
    severity: "high",
  },
  {
    id: 2,
    company: "Meta Platforms",
    type: "negative",
    message: "Regulatory concerns causing 9.4% score decline",
    time: "5 min ago",
    severity: "critical",
  },
  {
    id: 3,
    company: "Apple Inc.",
    type: "positive",
    message: "Strong earnings report drives 12.5% increase",
    time: "12 min ago",
    severity: "medium",
  },
  {
    id: 4,
    company: "Netflix Inc.",
    type: "negative",
    message: "Subscriber decline impacts credit score by -7.8%",
    time: "18 min ago",
    severity: "medium",
  },
  {
    id: 5,
    company: "Microsoft Corp.",
    type: "positive",
    message: "Cloud growth momentum boosts score by 8.3%",
    time: "25 min ago",
    severity: "low",
  },
]

export function AlertsFeed({ onCompanyClick }: AlertsFeedProps) {
  const [mutedAlerts, setMutedAlerts] = useState<Set<number>>(new Set())

  const handleShareAlert = (alert: any) => {
    // Generate PDF summary or share functionality
    console.log("[v0] Sharing alert:", alert.company, alert.message)
    // In real implementation, this would generate a PDF or open share dialog
  }

  const handleCreateTask = (alert: any) => {
    // Integrate with project management tools
    console.log("[v0] Creating task for:", alert.company, alert.message)
    // In real implementation, this would integrate with Jira/Asana
  }

  const handleMuteAlert = (alertId: number) => {
    setMutedAlerts((prev) => new Set([...prev, alertId]))
    console.log("[v0] Muted alert for 24h:", alertId)
    // In real implementation, this would set a 24h mute timer
  }

  return (
    <Card className="h-fit">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-primary" />
          Live Alerts
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {alerts
            .filter((alert) => !mutedAlerts.has(alert.id))
            .map((alert) => (
              <div key={alert.id} className="p-3 rounded-lg border border-border hover:bg-muted/50 transition-colors">
                <div className="flex items-start gap-2">
                  <div className="flex-shrink-0 mt-1">
                    {alert.type === "positive" ? (
                      <TrendingUp className="w-4 h-4 text-primary" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-destructive" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <button
                        onClick={() => onCompanyClick(alert.company)}
                        className="font-medium text-sm hover:text-primary transition-colors"
                      >
                        {alert.company}
                      </button>
                      <Badge
                        variant={
                          alert.severity === "critical"
                            ? "destructive"
                            : alert.severity === "high"
                              ? "default"
                              : "secondary"
                        }
                        className="text-xs"
                      >
                        {alert.severity}
                      </Badge>
                    </div>
                    <p className="text-xs text-muted-foreground mb-2">{alert.message}</p>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1 text-xs text-muted-foreground">
                        <Clock className="w-3 h-3" />
                        {alert.time}
                      </div>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                            <MoreHorizontal className="w-3 h-3" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-48">
                          <DropdownMenuItem onClick={() => handleShareAlert(alert)}>
                            <Share className="w-4 h-4 mr-2" />
                            Share/Export PDF
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => handleCreateTask(alert)}>
                            <Plus className="w-4 h-4 mr-2" />
                            Create Task
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => handleMuteAlert(alert.id)}>
                            <VolumeX className="w-4 h-4 mr-2" />
                            Mute for 24h
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </div>
                </div>
              </div>
            ))}
        </div>
      </CardContent>
    </Card>
  )
}
