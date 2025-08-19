"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Bell, X, Check, AlertTriangle, TrendingUp, TrendingDown, Clock, Mail, Phone, Monitor } from "lucide-react"

interface Notification {
  id: number
  title: string
  message: string
  type: "alert" | "info" | "warning" | "success"
  time: string
  read: boolean
  deliveryMethod: "email" | "call" | "notification"
  company?: string
}

interface NotificationsPanelProps {
  isOpen: boolean
  onClose: () => void
}

export function NotificationsPanel({ isOpen, onClose }: NotificationsPanelProps) {
  const [notifications, setNotifications] = useState<Notification[]>([
    {
      id: 1,
      title: "Critical Alert: Meta Platforms",
      message: "Regulatory concerns causing 9.4% score decline",
      type: "alert",
      time: "5 min ago",
      read: false,
      deliveryMethod: "notification",
      company: "Meta Platforms",
    },
    {
      id: 2,
      title: "Score Increase: Tesla Inc.",
      message: "Score increased 15% due to production milestone achievement",
      type: "success",
      time: "2 min ago",
      read: false,
      deliveryMethod: "notification",
      company: "Tesla Inc.",
    },
    {
      id: 3,
      title: "Alert Delivered via Phone",
      message: "Meta Platforms regulatory alert was delivered via phone call",
      type: "info",
      time: "6 min ago",
      read: true,
      deliveryMethod: "call",
    },
    {
      id: 4,
      title: "Email Alert Sent",
      message: "Apple Inc. score decrease alert sent to your email",
      type: "info",
      time: "2 hours ago",
      read: true,
      deliveryMethod: "email",
    },
    {
      id: 5,
      title: "Positive Movement: Apple Inc.",
      message: "Strong earnings report drives 12.5% increase",
      type: "success",
      time: "12 min ago",
      read: false,
      deliveryMethod: "notification",
      company: "Apple Inc.",
    },
    {
      id: 6,
      title: "Score Decline: Netflix Inc.",
      message: "Subscriber decline impacts credit score by -7.8%",
      type: "warning",
      time: "18 min ago",
      read: true,
      deliveryMethod: "notification",
      company: "Netflix Inc.",
    },
  ])

  const markAsRead = (id: number) => {
    setNotifications(notifications.map((notif) => (notif.id === id ? { ...notif, read: true } : notif)))
  }

  const markAllAsRead = () => {
    setNotifications(notifications.map((notif) => ({ ...notif, read: true })))
  }

  const deleteNotification = (id: number) => {
    setNotifications(notifications.filter((notif) => notif.id !== id))
  }

  const unreadCount = notifications.filter((n) => !n.read).length

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case "alert":
        return <AlertTriangle className="w-4 h-4 text-destructive" />
      case "success":
        return <TrendingUp className="w-4 h-4 text-primary" />
      case "warning":
        return <TrendingDown className="w-4 h-4 text-orange-500" />
      default:
        return <Bell className="w-4 h-4 text-muted-foreground" />
    }
  }

  const getDeliveryIcon = (method: string) => {
    switch (method) {
      case "email":
        return <Mail className="w-3 h-3" />
      case "call":
        return <Phone className="w-3 h-3" />
      case "notification":
        return <Monitor className="w-3 h-3" />
      default:
        return null
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-start justify-end">
      <div className="bg-background w-96 h-full shadow-xl border-l border-border">
        <Card className="h-full rounded-none border-0">
          <CardHeader className="border-b border-border">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Bell className="w-5 h-5 text-primary" />
                Notifications
                {unreadCount > 0 && (
                  <Badge variant="destructive" className="text-xs">
                    {unreadCount}
                  </Badge>
                )}
              </CardTitle>
              <Button variant="ghost" size="sm" onClick={onClose}>
                <X className="w-4 h-4" />
              </Button>
            </div>
            {unreadCount > 0 && (
              <Button variant="outline" size="sm" onClick={markAllAsRead} className="w-fit bg-transparent">
                <Check className="w-4 h-4 mr-2" />
                Mark all as read
              </Button>
            )}
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[calc(100vh-120px)]">
              <div className="p-4 space-y-3">
                {notifications.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Bell className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>No notifications</p>
                  </div>
                ) : (
                  notifications.map((notification) => (
                    <div
                      key={notification.id}
                      className={`p-3 rounded-lg border transition-colors ${
                        notification.read ? "bg-muted/30 border-border" : "bg-background border-primary/20 shadow-sm"
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 mt-1">{getNotificationIcon(notification.type)}</div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <h4
                              className={`text-sm font-medium ${!notification.read ? "text-foreground" : "text-muted-foreground"}`}
                            >
                              {notification.title}
                            </h4>
                            {!notification.read && <div className="w-2 h-2 bg-primary rounded-full flex-shrink-0" />}
                          </div>
                          <p className="text-xs text-muted-foreground mb-2 line-clamp-2">{notification.message}</p>
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <Clock className="w-3 h-3" />
                              {notification.time}
                              <div className="flex items-center gap-1">
                                {getDeliveryIcon(notification.deliveryMethod)}
                                <span className="capitalize">{notification.deliveryMethod}</span>
                              </div>
                            </div>
                            <div className="flex gap-1">
                              {!notification.read && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => markAsRead(notification.id)}
                                  className="h-6 px-2 text-xs"
                                >
                                  <Check className="w-3 h-3" />
                                </Button>
                              )}
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => deleteNotification(notification.id)}
                                className="h-6 px-2 text-xs text-destructive hover:text-destructive"
                              >
                                <X className="w-3 h-3" />
                              </Button>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
