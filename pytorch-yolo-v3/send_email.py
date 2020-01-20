import smtplib
import ssl
from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart  # New line
from email.mime.base import MIMEBase  # New line
from email import encoders  # New line

from settings import *

from threading import Thread


def threading(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
    return wrapper


@threading
def send_mail(sender_email, sender_name, password, receiver_emails, receiver_names, email_body, filename):

    for receiver_email, receiver_name in zip(receiver_emails, receiver_names):
        print("Sending the email...\n")
        # Configurating user's info
        msg = MIMEMultipart()
        msg['To'] = formataddr((receiver_name, receiver_email))
        msg['From'] = formataddr((sender_name, sender_email))
        msg['Subject'] = 'Hello, my friend ' + receiver_name

        msg.attach(MIMEText(email_body, 'html'))

        try:
            # Open PDF file in binary mode
            with open(filename, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())

            # Encode file in ASCII characters to send by email
            encoders.encode_base64(part)

            # Add header as key/value pair to attachment part
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {filename}",
            )

            msg.attach(part)
        except Exception as e:
            print(f'Oh no! We didn''t found the attachment!n{e}')
            break

        try:
            # Creating a SMTP session | use 587 with TLS, 465 SSL and 25
            server = smtplib.SMTP('smtp.gmail.com', 587)
            # Encrypts the email
            context = ssl.create_default_context()
            server.starttls(context=context)
            # We log in into our Google account
            server.login(sender_email, password)
            # Sending email from sender, to receiver with the email body
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print('Email sent!')
        except Exception as e:
            print(f'Oh no! Something bad happened!n{e}')
            break
        finally:
            print('Closing the server...')
            server.quit()


if __name__ == '__main__':

    # User configuration
    # sender_email = 'pavel.semin.USA@gmail.com'
    # sender_name = 'pasha_s_sender'
    # password = input('Please, type your password:n')

    # receiver_emails = ['uzerp2apple@gmail.com']
    # receiver_names = ['pasha_s_receiver']

    sender_email = SENDER_EMAIL
    sender_name = SENDER_NAME
    password = PASSWORD
    # password = input('Please, type your password:n')

    receiver_emails = RECEIVER_EMAILS
    receiver_names = RECEIVER_NAMES

    # Email body
    email_html = open('email.html')
    email_body = email_html.read()

    filename = './imgs/dog.jpg'
    # filename = 'document.pdf'

    send_mail(sender_email, sender_name, password, receiver_emails,
              receiver_names, email_body, filename)

    print('____________________________________')
