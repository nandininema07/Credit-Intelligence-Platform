import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { TrendingUp, Search } from "lucide-react"
import { MarketPulseChart } from "@/components/market-pulse-chart"
import { TopMoversCard } from "@/components/top-movers-card"
import { WatchlistCard } from "@/components/watchlist-card"
import { AlertsFeed } from "@/components/alerts-feed"

interface MainDashboardProps {
  onCompanyClick: (companyName: string) => void
}

export function MainDashboard({ onCompanyClick }: MainDashboardProps) {
  return (
    <div className="p-6 space-y-6 overflow-auto h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-playfair text-3xl font-bold text-foreground">Market Overview</h1>
          <p className="text-muted-foreground">Real-time credit risk monitoring across global markets</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input placeholder="Search companies..." className="pl-10 w-80" />
          </div>
          <Badge variant="outline" className="text-primary border-primary">
            <div className="w-2 h-2 bg-primary rounded-full mr-2 animate-pulse" />
            Live Data
          </Badge>
        </div>
      </div>

      {/* Market Pulse */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary" />
            Market Pulse
          </CardTitle>
        </CardHeader>
        <CardContent>
          <MarketPulseChart />
        </CardContent>
      </Card>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Top Movers */}
        <div className="lg:col-span-2 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <TopMoversCard title="Top Risers" type="positive" onCompanyClick={onCompanyClick} />
            <TopMoversCard title="Top Fallers" type="negative" onCompanyClick={onCompanyClick} />
          </div>

          {/* Watchlist */}
          <WatchlistCard onCompanyClick={onCompanyClick} />
        </div>

        {/* Alerts Feed */}
        <div>
          <AlertsFeed onCompanyClick={onCompanyClick} />
        </div>
      </div>
    </div>
  )
}
