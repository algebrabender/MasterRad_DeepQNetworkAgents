import win32com.client as win32

def send_mail(msg, subject='Training has finished'):
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.Subject = 'Training has finished'
    mail.To = "jelena.firulovic@gmail.com"
    mail.HTMLBody = msg

    mail.Send()