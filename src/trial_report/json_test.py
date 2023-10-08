from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import *
from reportlab.lib.pagesizes import A4

path = "report.pdf"
a = SimpleDocTemplate(
    path,
    pagesize=A4,
    rightMargin=72,
    leftMargin=72,
    topMargin=72,
    bottomMargin=18)

story = []
paragraph_title = Paragraph("title")
for _ in range(100):
    story.append(paragraph_title)
a.build(story)

