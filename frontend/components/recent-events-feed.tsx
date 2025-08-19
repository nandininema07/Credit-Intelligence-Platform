import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { FileText, MessageSquare, TrendingUp, TrendingDown, ExternalLink } from "lucide-react"

interface RecentEventsFeedProps {
  company: string
}

const events = [
  {
    id: 1,
    type: "news",
    title: "Q4 Earnings Report Exceeds Expectations",
    source: "Financial Times",
    sentiment: "positive",
    impact: "+8.5",
    time: "2 hours ago",
    summary: "Company reported record quarterly revenue with strong growth across all segments.",
  },
  {
    id: 2,
    type: "social",
    title: "CEO Interview on Future Strategy",
    source: "Twitter/X",
    sentiment: "positive",
    impact: "+2.1",
    time: "4 hours ago",
    summary: "Positive market reaction to CEO comments on expansion plans and innovation roadmap.",
  },
  {
    id: 3,
    type: "filing",
    title: "SEC Form 10-K Annual Report Filed",
    source: "SEC EDGAR",
    sentiment: "neutral",
    impact: "0.0",
    time: "1 day ago",
    summary: "Annual report filed with standard disclosures and risk factors.",
  },
  {
    id: 4,
    type: "news",
    title: "Supply Chain Disruption Concerns",
    source: "Reuters",
    sentiment: "negative",
    impact: "-2.8",
    time: "2 days ago",
    summary: "Analysts express concerns about potential supply chain impacts on production.",
  },
]

export function RecentEventsFeed({ company }: RecentEventsFeedProps) {
  return (
    <div className="space-y-4">
      {events.map((event) => (
        <div key={event.id} className="p-4 rounded-lg border border-border hover:bg-muted/50 transition-colors">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-1">
              {event.type === "news" ? (
                <FileText className="w-5 h-5 text-primary" />
              ) : event.type === "social" ? (
                <MessageSquare className="w-5 h-5 text-accent" />
              ) : (
                <FileText className="w-5 h-5 text-muted-foreground" />
              )}
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h4 className="font-medium text-sm mb-1">{event.title}</h4>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>{event.source}</span>
                    <span>•</span>
                    <span>{event.time}</span>
                  </div>
                </div>

                <div className="flex items-center gap-2 ml-4">
                  <Badge
                    variant={
                      event.sentiment === "positive"
                        ? "default"
                        : event.sentiment === "negative"
                          ? "destructive"
                          : "secondary"
                    }
                    className="text-xs"
                  >
                    {event.sentiment === "positive" ? (
                      <TrendingUp className="w-3 h-3 mr-1" />
                    ) : event.sentiment === "negative" ? (
                      <TrendingDown className="w-3 h-3 mr-1" />
                    ) : null}
                    {event.impact}
                  </Badge>
                  <Button size="sm" variant="ghost">
                    <ExternalLink className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              <p className="text-sm text-muted-foreground">{event.summary}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
