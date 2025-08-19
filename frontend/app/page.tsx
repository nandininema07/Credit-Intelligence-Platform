"use client"

import { useState } from "react"
import { Sidebar } from "@/components/sidebar"
import { MainDashboard } from "@/components/main-dashboard"
import { CompanyDeepDive } from "@/components/company-deep-dive"
import { ChatbotTab } from "@/components/chatbot-tab"
import { AlertsConfig } from "@/components/alerts-config"
import { CompanyModal } from "@/components/company-modal"
import { NotificationsPanel } from "@/components/notifications-panel"

export default function HomePage() {
  const [activeTab, setActiveTab] = useState("dashboard")
  const [selectedCompany, setSelectedCompany] = useState<string | null>(null)
  const [modalCompany, setModalCompany] = useState<string | null>(null)
  const [notificationsPanelOpen, setNotificationsPanelOpen] = useState(false)

  const handleCompanyClick = (companyName: string) => {
    setModalCompany(companyName)
  }

  const closeModal = () => {
    setModalCompany(null)
  }

  const toggleNotificationsPanel = () => {
    setNotificationsPanelOpen(!notificationsPanelOpen)
  }

  return (
    <div className="flex h-screen bg-background">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} onNotificationsClick={toggleNotificationsPanel} />

      <main className="flex-1 overflow-hidden">
        {activeTab === "dashboard" && <MainDashboard onCompanyClick={handleCompanyClick} />}
        {activeTab === "deep-dive" && (
          <CompanyDeepDive selectedCompany={selectedCompany} onCompanyChange={setSelectedCompany} />
        )}
        {activeTab === "chatbot" && <ChatbotTab />}
        {activeTab === "alerts" && <AlertsConfig />}
      </main>

      {modalCompany && <CompanyModal companyName={modalCompany} onClose={closeModal} />}

      <NotificationsPanel isOpen={notificationsPanelOpen} onClose={() => setNotificationsPanelOpen(false)} />
    </div>
  )
}
