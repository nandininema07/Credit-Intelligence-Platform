"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { MessageSquare, Send, Bot, User, Lightbulb } from "lucide-react"

interface Message {
  id: number
  type: "user" | "bot"
  content: string
  timestamp: Date
}

const suggestedQuestions = [
  "Why did Apple's score change today?",
  "Compare Tesla and Ford credit risk",
  "What are the key risks for Meta?",
  "Show me companies with improving scores",
  "Explain the SHAP analysis for Microsoft",
]

export function ChatbotTab() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      type: "bot",
      content:
        "Hello! I'm your AI assistant for credit risk analysis. I can help you understand score changes, compare companies, and explain our AI insights. What would you like to know?",
      timestamp: new Date(),
    },
  ])
  const [inputValue, setInputValue] = useState("")

  const handleSendMessage = () => {
    if (!inputValue.trim()) return

    const userMessage: Message = {
      id: messages.length + 1,
      type: "user",
      content: inputValue,
      timestamp: new Date(),
    }

    // Simulate bot response
    const botResponse: Message = {
      id: messages.length + 2,
      type: "bot",
      content: generateBotResponse(inputValue),
      timestamp: new Date(),
    }

    setMessages([...messages, userMessage, botResponse])
    setInputValue("")
  }

  const generateBotResponse = (question: string): string => {
    if (question.toLowerCase().includes("apple")) {
      return "Apple's credit score increased by 12.5% today primarily due to their strong Q4 earnings report. The key drivers were: 1) Revenue beat expectations by 8%, 2) iPhone sales exceeded forecasts, and 3) Services revenue grew 15% year-over-year. The SHAP analysis shows earnings performance contributed +8.5 points to the score."
    }
    if (question.toLowerCase().includes("compare")) {
      return "I can help you compare companies! For a detailed comparison, I'll need to know which specific companies you'd like to analyze. I can compare their current scores, recent trends, key risk factors, and provide insights on which might be a better credit risk."
    }
    return "I understand you're asking about credit risk analysis. Let me help you with that. Could you be more specific about which company or aspect you'd like to explore? I can provide detailed explanations about score changes, risk factors, or comparative analysis."
  }

  const handleSuggestedQuestion = (question: string) => {
    setInputValue(question)
  }

  return (
    <div className="p-6 h-full flex flex-col">
      <div className="mb-6">
        <h1 className="font-playfair text-3xl font-bold text-foreground">AI Assistant</h1>
        <p className="text-muted-foreground">Ask questions about credit risk data in natural language</p>
      </div>

      <div className="flex-1 flex gap-6">
        {/* Chat Interface */}
        <div className="flex-1 flex flex-col">
          <Card className="flex-1 flex flex-col">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="w-5 h-5 text-primary" />
                Chat with AI Assistant
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              {/* Messages */}
              <div className="flex-1 overflow-y-auto space-y-4 mb-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-3 ${message.type === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {message.type === "bot" && (
                      <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
                        <Bot className="w-4 h-4 text-primary-foreground" />
                      </div>
                    )}
                    <div
                      className={`max-w-[80%] p-3 rounded-lg ${
                        message.type === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                      }`}
                    >
                      <p className="text-sm">{message.content}</p>
                      <p className="text-xs opacity-70 mt-1">{message.timestamp.toLocaleTimeString()}</p>
                    </div>
                    {message.type === "user" && (
                      <div className="w-8 h-8 bg-secondary rounded-full flex items-center justify-center flex-shrink-0">
                        <User className="w-4 h-4 text-secondary-foreground" />
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Input */}
              <div className="flex gap-2">
                <Input
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Ask about credit risk, company analysis, or score explanations..."
                  onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
                  className="flex-1"
                />
                <Button onClick={handleSendMessage}>
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Suggested Questions Sidebar */}
        <div className="w-80">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lightbulb className="w-5 h-5 text-primary" />
                Suggested Questions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {suggestedQuestions.map((question, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    className="w-full text-left justify-start h-auto p-3 bg-transparent"
                    onClick={() => handleSuggestedQuestion(question)}
                  >
                    <span className="text-sm">{question}</span>
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="text-sm">Quick Actions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                  Generate Risk Report
                </Button>
                <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                  Compare Top 5 Companies
                </Button>
                <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                  Explain Today's Alerts
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
