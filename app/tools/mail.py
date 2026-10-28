"""Email to saved contacts over SMTP (moved here from Saturday). Contacts live in data/contacts.json:
{"contacts": [{"name": "alex", "email": "alex@example.com"}]}; SMTP_EMAIL / SMTP_PASSWORD (and optionally
SMTP_HOST / SMTP_PORT) in .env."""
import json
import os
import smtplib
from email.message import EmailMessage

from app.tools.base import Plugin
from constants import CONTACTS_PATH
from saturday.utils import require_env


def load_contacts(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f)['contacts']


def find_contact(name, field, path=CONTACTS_PATH):
    """Match a spoken name against contacts.json; exact name first, then any contact whose name contains it."""
    wanted = name.lower().strip()
    contacts = [c for c in load_contacts(path) if c.get(field)]
    for contact in contacts:
        if contact['name'].lower() == wanted:
            return contact
    for contact in contacts:
        if wanted in contact['name'].lower() or contact['name'].lower() in wanted:
            return contact
    known = ', '.join(c['name'] for c in contacts) or 'none'
    raise ValueError(f"no {field} for '{name}'; known contacts: {known}")


class SendEmail(Plugin):

    name = 'send_email'
    description = 'Send an email to a saved contact.'
    parameters = {'contact': 'contact name', 'subject': 'subject line', 'body': 'email text'}
    required = ('contact', 'body')
    confirm = True

    def summary(self, arguments):
        return f"email {arguments.get('contact', '')} about \"{arguments.get('subject') or arguments.get('body', '')[:40]}\""

    def run(self, contact, body, subject=''):
        person = find_contact(contact, 'email', self.config.path(CONTACTS_PATH))
        subject = subject or f"Message from {self.config.name}"
        sender, password = require_env('SMTP_EMAIL'), require_env('SMTP_PASSWORD')
        host, port = os.environ.get('SMTP_HOST', 'smtp.gmail.com'), int(os.environ.get('SMTP_PORT', '587'))
        mail = EmailMessage()
        mail['From'], mail['To'], mail['Subject'] = sender, person['email'], subject
        mail.set_content(body)
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(sender, password)
            smtp.send_message(mail)
        return f"emailed {person['name']} ({person['email']}): {subject}"
