"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { Bell, Plus, Trash2, Mail, Phone, Monitor } from "lucide-react"

interface Alert {
  id: number
  name: string
  company: string
  threshold: string
  condition: string
  delivery: string[]
  enabled: boolean
}

export function AlertsConfig() {
  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: 1,
      name: "Apple High Risk Alert",
      company: "Apple Inc.",
      threshold: "5%",
      condition: "score_decrease",
      delivery: ["email", "notification"],
      enabled: true,
    },
    {
      id: 2,
      name: "Tesla Volatility Monitor",
      company: "Tesla Inc.",
      threshold: "10%",
      condition: "score_change",
      delivery: ["notification"],
      enabled: true,
    },
    {
      id: 3,
      name: "Meta Regulatory Watch",
      company: "Meta Platforms",
      threshold: "3%",
      condition: "keyword_mention",
      delivery: ["email", "call", "notification"],
      enabled: false,
    },
  ])

  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newAlertDelivery, setNewAlertDelivery] = useState<string[]>([])

  const toggleAlert = (id: number) => {
    setAlerts(alerts.map((alert) => (alert.id === id ? { ...alert, enabled: !alert.enabled } : alert)))
  }

  const deleteAlert = (id: number) => {
    setAlerts(alerts.filter((alert) => alert.id !== id))
  }

  const getDeliveryIcons = (delivery: string[]) => {
    return delivery.map((method) => {
      switch (method) {
        case "email":
          return <Mail key={method} className="w-4 h-4" />
        case "call":
          return <Phone key={method} className="w-4 h-4" />
        case "notification":
          return <Monitor key={method} className="w-4 h-4" />
        default:
          return null
      }
    })
  }

  const handleDeliveryChange = (method: string, checked: boolean) => {
    if (checked) {
      setNewAlertDelivery([...newAlertDelivery, method])
    } else {
      setNewAlertDelivery(newAlertDelivery.filter((m) => m !== method))
    }
  }

  return (
    <div className="p-6 space-y-6 overflow-auto h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-playfair text-3xl font-bold text-foreground">Alert Configuration</h1>
          <p className="text-muted-foreground">Set up custom alerts for proactive risk monitoring</p>
        </div>
        <Button onClick={() => setShowCreateForm(true)}>
          <Plus className="w-4 h-4 mr-2" />
          Create Alert
        </Button>
      </div>

      {/* Create Alert Form */}
      {showCreateForm && (
        <Card>
          <CardHeader>
            <CardTitle>Create New Alert</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="alert-name">Alert Name</Label>
                <Input id="alert-name" placeholder="Enter alert name" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="company-select">Company</Label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select company" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="apple">Apple Inc.</SelectItem>
                    <SelectItem value="microsoft">Microsoft Corp.</SelectItem>
                    <SelectItem value="tesla">Tesla Inc.</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="threshold">Threshold</Label>
                <Input id="threshold" placeholder="e.g., 5%" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="condition">Condition</Label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select condition" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="score_increase">Score Increase</SelectItem>
                    <SelectItem value="score_decrease">Score Decrease</SelectItem>
                    <SelectItem value="score_change">Any Score Change</SelectItem>
                    <SelectItem value="keyword_mention">Keyword Mention</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="mt-4 space-y-3">
              <Label>Delivery Methods</Label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex items-center space-x-2 p-3 border rounded-lg">
                  <Checkbox
                    id="email"
                    checked={newAlertDelivery.includes("email")}
                    onCheckedChange={(checked) => handleDeliveryChange("email", checked as boolean)}
                  />
                  <Mail className="w-4 h-4 text-muted-foreground" />
                  <div>
                    <Label htmlFor="email" className="text-sm font-medium">
                      Email
                    </Label>
                    <p className="text-xs text-muted-foreground">Send alerts via email</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2 p-3 border rounded-lg">
                  <Checkbox
                    id="call"
                    checked={newAlertDelivery.includes("call")}
                    onCheckedChange={(checked) => handleDeliveryChange("call", checked as boolean)}
                  />
                  <Phone className="w-4 h-4 text-muted-foreground" />
                  <div>
                    <Label htmlFor="call" className="text-sm font-medium">
                      Phone Call
                    </Label>
                    <p className="text-xs text-muted-foreground">Receive phone call alerts</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2 p-3 border rounded-lg">
                  <Checkbox
                    id="notification"
                    checked={newAlertDelivery.includes("notification")}
                    onCheckedChange={(checked) => handleDeliveryChange("notification", checked as boolean)}
                  />
                  <Monitor className="w-4 h-4 text-muted-foreground" />
                  <div>
                    <Label htmlFor="notification" className="text-sm font-medium">
                      In-App Notification
                    </Label>
                    <p className="text-xs text-muted-foreground">Show in notification panel</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex gap-2 mt-6">
              <Button>Create Alert</Button>
              <Button variant="outline" onClick={() => setShowCreateForm(false)}>
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Active Alerts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-primary" />
            Active Alerts ({alerts.filter((a) => a.enabled).length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {alerts.map((alert) => (
              <div key={alert.id} className="flex items-center justify-between p-4 rounded-lg border border-border">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="font-medium">{alert.name}</h3>
                    <Badge variant={alert.enabled ? "default" : "secondary"}>
                      {alert.enabled ? "Active" : "Disabled"}
                    </Badge>
                  </div>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>Company: {alert.company}</p>
                    <p>
                      Trigger: {alert.condition.replace("_", " ")} by {alert.threshold}
                    </p>
                    <div className="flex items-center gap-2">
                      <span>Delivery:</span>
                      <div className="flex gap-1">{getDeliveryIcons(alert.delivery)}</div>
                      <span className="text-xs">
                        {alert.delivery
                          .map((method) => {
                            switch (method) {
                              case "email":
                                return "Email"
                              case "call":
                                return "Call"
                              case "notification":
                                return "In-App"
                              default:
                                return method
                            }
                          })
                          .join(", ")}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Switch checked={alert.enabled} onCheckedChange={() => toggleAlert(alert.id)} />
                  <Button variant="ghost" size="sm" onClick={() => deleteAlert(alert.id)}>
                    <Trash2 className="w-4 h-4 text-destructive" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Alert History */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Alert History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[
              {
                time: "2 hours ago",
                message: "Apple Inc. score decreased by 5.2%",
                type: "warning",
                delivery: "Email, In-App",
              },
              {
                time: "4 hours ago",
                message: "Tesla Inc. volatility alert triggered",
                type: "info",
                delivery: "In-App",
              },
              {
                time: "1 day ago",
                message: 'Meta Platforms keyword mention: "regulation"',
                type: "alert",
                delivery: "Email, Call, In-App",
              },
              {
                time: "2 days ago",
                message: "Microsoft Corp. score increased by 8.3%",
                type: "success",
                delivery: "Email, In-App",
              },
            ].map((item, index) => (
              <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                <div className="flex-1">
                  <p className="text-sm font-medium">{item.message}</p>
                  <div className="flex items-center gap-4 mt-1">
                    <p className="text-xs text-muted-foreground">{item.time}</p>
                    <p className="text-xs text-muted-foreground">Sent via: {item.delivery}</p>
                  </div>
                </div>
                <Badge
                  variant={item.type === "warning" ? "destructive" : item.type === "success" ? "default" : "secondary"}
                >
                  {item.type}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
