"""
email_client.py — Gmail SMTP helper for sending plain-text emails.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def send_email(to_address: str, subject: str, body: str) -> bool:
    """
    Send a plain-text email via Gmail SMTP.

    Args:
        to_address: recipient email address
        subject: email subject line
        body: plain-text email body

    Returns:
        True if sent successfully, False otherwise.
    """
    msg = MIMEMultipart()
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_address
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, to_address, msg.as_string())
        print(f"[EMAIL] Sent to {to_address}: {subject}")
        return True
    except Exception as exc:
        print(f"[EMAIL ERROR] Failed to send to {to_address}: {exc}")
        return False
