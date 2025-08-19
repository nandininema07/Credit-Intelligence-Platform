# Imports
import logging
import smtplib
from typing import Dict, List, Any
from datetime import datetime

class AlertManager:
    @staticmethod
    # Function: send_pipeline_alert()
    def send_pipeline_alert(config: Dict, message: str) -> None:
        try:
            with smtplib.SMTP(config['email']['smtp_server'], config['email']['smtp_port']) as server:
                server.starttls()
                server.login(config['email']['username'], config['email']['password'])
                server.sendmail(
                    config['email']['from_address'],
                    config['email']['to_addresses'],
                    f"Subject: Pipeline Alert\n\n{message}"
                )
            logging.info("Alert sent successfully")
        except Exception as e:
            logging.error(f"Failed to send alert: {e}")

    @staticmethod
    # Function: check_error_thresholds()
    def check_error_thresholds(metrics: Dict[str, Any], config: Dict) -> None:
        error_count = metrics.get("errors", 0)
        if error_count > config['alerting']['error_threshold']:
            AlertManager.send_pipeline_alert(config, f"Error threshold exceeded: {error_count} errors")