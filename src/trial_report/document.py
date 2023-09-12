from reportlab.pdfgen import canvas
from reportlab.pdfbase import *
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import *
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
import numpy as np
from dataclasses import dataclass


@dataclass
class ClinicalAnalysis:
    limb_length: dict
    grf_max: list
    grf_impulse: list
    toe_clearance: list
    stance_time: list
    gait_speed_imu: list
    gait_speed_distance: list
    subject_name: str = "Name"
    age: str = str(30)
    weight: str = str(70.0)
    test_date: str = "2023-01-01"
    test_label: str = "RH-23-N/A"
    session_type: str = "10m Cue on"
    paretic_side: str = "LEFT"
    sole_size: str = str(270)
    sensor_calibration_date: str = "2023-07-01"


def num_array_to_string_array(num_array):
    array_return = [""] * len(num_array)
    for i in range(len(num_array)):
        array_return[i] = str(num_array[i])

    return array_return


def make_report(path, data_report: ClinicalAnalysis):
    # Document Formatting
    story = []

    doc = SimpleDocTemplate(
        path+"/analysis report.pdf",
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18)

    styles = getSampleStyleSheet()

    style_sub1_table = TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('LINEAFTER', (0, 0), (0, -1), 0.5, colors.gray),
        ('LINEAFTER', (2, 0), (2, -2), 0.5, colors.gray),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (2, 0), (2, -2), colors.lightgrey),
        ('LINEABOVE', (0, 0), (-1, 0), 2, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 2, colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (2, 0), (2, 5), 'LEFT'),
        ('ALIGN', (1, 0), (1, 5), 'RIGHT'),
        ('ALIGN', (3, 0), (3, 5), 'RIGHT'),
        ('VALIGN', (0,0), (-1, -1), 'MIDDLE'),
        ('RIGHTPADDING', (1, 0), (1, 5), 20),
        ('LEFTPADDING', (2, 0), (2, 5), 20)
    ])

    style_sub2_table = TableStyle([
        ('FONTSIZE', (0,0), (-1, -1), 11),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (2, 1), (2, -1), 'RIGHT'),
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ('ALIGN', (3, 1), (3, -1), 'RIGHT'),
        ('ALIGN', (0,0), (-1, 0), 'CENTER'),
        ('LINEAFTER', (0, 0), (-2, -1), 0.5, colors.gray),
        ('VALIGN', (0,0), (-1, -1), 'MIDDLE'),
        ('RIGHTPADDING', (1,0), (-1, -1), 10),
        ('LINEABOVE', (0, 0), (-1, 0), 2, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 2, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.gray),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (0,0), (-1, 0), colors.lightgrey)
    ])

    title = "Training Report"

    sub1_title = "1. Basic info"
    label_sub1_left = ["Subject Name",
                       "Age",
                       "Weight",
                       "Paretic side",
                       "Sole size"]
    label_sub1_right = ["Test Date",
                        "Test Label",
                        "Session Type",
                        "Cue"
                        "Sensor Calibration date", ""]

    sub3_title = "3. Limb Length"
    label_sub3_column = ["Femur", "Tibia", "Foot", "Pelvis"]

    sub2_title = "2. Symmetry Analysis"
    label_sub2_column = ["Mean", "Std", "Symmetry"]
    label_sub2_row = [
        "GRF max [N]",
        "GRF Impulse",
        "Toe Clearance",
        "Stance Time",
        "Gait Speed (IMU)",
        "Gait Speed (Distance)"
    ]

    subject_name = data_report.subject_name
    age = data_report.age
    weight = data_report.weight
    test_date = data_report.test_date
    test_label = data_report.test_label
    session_type = data_report.session_type
    paretic_side = data_report.paretic_side
    sole_size = data_report.sole_size
    sensor_calibration_date = data_report.sensor_calibration_date

    data_sub1_left = [subject_name,
                      age,
                      weight,
                      paretic_side,
                      sole_size]
    data_sub1_right = [test_date, test_label,
                       session_type, sensor_calibration_date,
                       None]

    limb_length_femur = data_report.limb_length["Femur"]
    limb_length_tibia = data_report.limb_length["Tibia"]
    limb_length_foot = data_report.limb_length["Foot"]
    limb_length_pelvis = data_report.limb_length["Pelvis"]

    limb_length = \
        [["Femur", limb_length_femur, "Tibia", limb_length_tibia],
         ["Foot", limb_length_foot, "Pelvis", limb_length_pelvis]]

    # Mean, stdev, symmetry order
    grf_max = num_array_to_string_array(data_report.grf_max)
    grf_impulse = num_array_to_string_array(data_report.grf_impulse)
    toe_clearance = num_array_to_string_array(data_report.toe_clearance)
    stance_time = num_array_to_string_array(data_report.stance_time)
    gait_speed_imu = num_array_to_string_array(data_report.gait_speed_imu)
    gait_speed_distance = \
        num_array_to_string_array(data_report.gait_speed_distance)

    paragraph_title = Paragraph(title, styles['Title'])
    paragraph_sub1_title = Paragraph(sub1_title, styles['Heading2'])
    paragraph_sub2_title = Paragraph(sub2_title, styles['Heading2'])
    paragraph_sub3_title = Paragraph(sub3_title, styles['Heading2'])

    text_sub1 = np.array(
        [label_sub1_left, data_sub1_left,
         label_sub1_right, data_sub1_right]
    )
    text_sub1 = text_sub1.transpose().tolist()
    table_sub1 = Table(text_sub1,
                       colWidths=(110, 75, 170, 85),
                       rowHeights=30,
                       style=style_sub1_table,)

    text_sub2 = [[""] + label_sub2_column,
                 ["GRF Max"] + grf_max,
                 ["GRF Impulse"] + grf_impulse,
                 ["Toe Clearance"] + toe_clearance,
                 ["Stance Time"] + stance_time,
                 ["Gait Speed (IMU)"] + gait_speed_imu,
                 ["Gait Speed (Distance)"] + gait_speed_distance]
    table_sub2 = Table(text_sub2,
                       style=style_sub2_table,
                       colWidths=(160, 95, 95, 95),
                       rowHeights=30)

    text_sub3 = limb_length
    table_sub3 = Table(text_sub3)

    spacer_10_point = Spacer(1, 10)
    spacer_30_point = Spacer(1, 30)

    story.append(paragraph_title)
    story.append(spacer_10_point)
    story.append(paragraph_sub1_title)
    story.append(spacer_10_point)
    story.append(table_sub1)
    story.append(spacer_30_point)
    story.append(paragraph_sub2_title)
    story.append(spacer_10_point)
    story.append(table_sub2)

    doc.build(story)
