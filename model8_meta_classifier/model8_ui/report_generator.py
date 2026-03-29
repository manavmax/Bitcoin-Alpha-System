from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

def generate_report(state, filename="Model8_Report.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h - 50, "Bitcoin Market Intelligence — Model 8")

    c.setFont("Helvetica", 12)
    y = h - 120

    for k, v in state.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 25

    c.drawString(50, 80, "Research system only. Not financial advice.")
    c.save()
