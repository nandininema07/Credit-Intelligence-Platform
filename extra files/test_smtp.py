import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

transporter = smtplib.SMTP(os.getenv('SMTP_HOST'), os.getenv('SMTP_PORT'))

transporter.starttls()

transporter.login(os.getenv('SMTP_USERNAME'), os.getenv('SMTP_PASSWORD'))

mailOptions = {
  'from': os.getenv('SMTP_USERNAME'),
  'to': 'nandini.semstack2327@gmail.com',
  'subject': 'Test Email',
  'text': 'Hello, this is a test email sent using Gmail SMTP!'

}

transporter.sendmail(mailOptions['from'], mailOptions['to'], mailOptions['text'])

transporter.quit()



