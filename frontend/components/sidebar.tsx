"use client"

import { cn } from "@/lib/utils"
import { BarChart3, Building2, MessageSquare, Bell, TrendingUp, Shield } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface SidebarProps {
  activeTab: string
  onTabChange: (tab: string) => void
  onNotificationsClick: () => void
}

const navigation = [
  { id: "dashboard", name: "Dashboard", icon: BarChart3 },
  { id: "deep-dive", name: "Company Analysis", icon: Building2 },
  { id: "chatbot", name: "AI Assistant", icon: MessageSquare },
  { id: "alerts", name: "Alert Settings", icon: Bell },
]

export function Sidebar({ activeTab, onTabChange, onNotificationsClick }: SidebarProps) {
  return (
    <div className="w-64 bg-sidebar border-r border-sidebar-border flex flex-col">
      <div className="p-6 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <Shield className="w-5 h-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="font-playfair font-bold text-lg text-sidebar-foreground">CreditRisk Pro</h1>
            <p className="text-xs text-muted-foreground">Financial Monitoring</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navigation.map((item) => {
            const Icon = item.icon
            return (
              <li key={item.id}>
                <button
                  onClick={() => onTabChange(item.id)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
                    activeTab === item.id
                      ? "bg-sidebar-primary text-sidebar-primary-foreground"
                      : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                  )}
                >
                  <Icon className="w-5 h-5" />
                  {item.name}
                </button>
              </li>
            )
          })}
        </ul>

        <div className="mt-6 pt-4 border-t border-sidebar-border">
          <Button
            variant="outline"
            onClick={onNotificationsClick}
            className="w-full justify-start gap-3 bg-transparent border-sidebar-border hover:bg-sidebar-accent"
          >
            <Bell className="w-5 h-5" />
            <span>Notifications</span>
            <Badge variant="destructive" className="ml-auto text-xs">
              3
            </Badge>
          </Button>
        </div>
      </nav>

      <div className="p-4 border-t border-sidebar-border">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <TrendingUp className="w-4 h-4 text-primary" />
          <span>Market Status: Active</span>
        </div>
      </div>
    </div>
  )
}
