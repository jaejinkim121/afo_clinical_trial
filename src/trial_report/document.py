from reportlab.platypus import *
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import numpy as np
from define import ClinicalDataset


def num_array_to_string_array(num_array):
    array_return = [""] * len(num_array)
    for i in range(len(num_array)):
        array_return[i] = "{:.3f}".format(num_array[i])

    return array_return


def make_report(path, data_report: ClinicalDataset):
    # Document Formatting
    story = []
    path_base = \
        "{}/report/data/{}/{}/".format(
            path,
            data_report.metadata.test_label,
            data_report.metadata.session_type
        )
    pdf_path = path_base + 'report.pdf'
    image_path_base = path_base + "graph/"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18)

    styles = getSampleStyleSheet()

    style_sub1_table = TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('LINEAFTER', (0, 0), (0, -1), 0.5, colors.gray),
        ('LINEAFTER', (2, 0), (2, -1), 0.5, colors.gray),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (2, 0), (2, -1), colors.lightgrey),
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
        ('FONTSIZE', (0,0), (-1, -1), 9),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('ALIGN', (0,0), (-1, 0), 'CENTER'),
        ('LINEAFTER', (0, 0), (-2, -1), 0.5, colors.gray),
        ('VALIGN', (0,0), (-1, -1), 'MIDDLE'),
        ('RIGHTPADDING', (1,1), (-1, -1), 10),
        ('LINEABOVE', (0, 0), (-1, 0), 2, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 2, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.gray),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (0,0), (-1, 0), colors.lightgrey)
    ])

    title = "Training Report"
    graph_title = "Data analysis"

    sub1_title = "1. Basic info"
    label_sub1_left = ["Subject Name",
                       "Age",
                       "Weight",
                       "Paretic side",
                       "Sole size"]
    label_sub1_right = ["Test Date",
                        "Test Label",
                        "Session Type",
                        "Cue",
                        "Sensor Calibration date"]

    sub3_title = "3. Limb Length"
    label_sub3_column = ["Femur", "Tibia", "Foot", "Pelvis"]

    sub2_title = "2. Symmetry Analysis"
    label_sub2_column = ["Paretic\nMean", "Paretic\nStd",
                         "Non-paretic\nMean", "Non-paretic\nStd",
                         "Symmetry"]
    label_sub2_row = [
        "GRF max [N]",
        "GRF Impulse",
        "Toe Clearance",
        "Stance Time",
        "Gait Speed (IMU)",
        "Gait Speed (Distance)"
    ]

    subject_name = data_report.metadata.name
    age = data_report.metadata.age
    weight = data_report.metadata.body_weight
    test_date = data_report.metadata.test_date
    test_label = data_report.metadata.test_label
    session = data_report.metadata.session_type.value
    session_list = session.split('_')
    session_type = session_list[0] + " " + session_list[1]
    paretic_side = data_report.metadata.paretic_side.value
    sole_size = data_report.metadata.sole_size
    sensor_calibration_date = data_report.metadata.date_calibration

    data_sub1_left = [subject_name,
                      age,
                      weight,
                      paretic_side,
                      sole_size]
    data_sub1_right = [test_date, test_label,
                       session_type, session[-1],
                       sensor_calibration_date]

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

    sole_sensor_text = "Data derived from sole sensor data"
    kinematics_text = "Kinematic data derived from IMU data"

    paragraph_title = Paragraph(title, styles['Title'])
    paragraph_sub1_title = Paragraph(sub1_title, styles['Heading2'])
    paragraph_sub2_title = Paragraph(sub2_title, styles['Heading2'])
    paragraph_sub3_title = Paragraph(sub3_title, styles['Heading2'])
    paragraph_graph_title = Paragraph(graph_title, styles['Title'])
    paragraph_graph_sole_sensor = Paragraph(sole_sensor_text, styles['Heading2'])
    paragraph_graph_kinematics = Paragraph(kinematics_text, styles['Heading2'])

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
                       colWidths=(130, 65, 65, 65, 65, 65),
                       rowHeights=30)

    text_sub3 = limb_length
    table_sub3 = Table(text_sub3)

    image_path_grf_impulse = image_path_base + "GRF_impulse_along_cycle.png"
    image_path_grf_max = image_path_base + "GRF_max_along_cycle.png"
    image_path_grf_mean_cycle = image_path_base + "GRF_mean_cycle.png"
    image_path_stance_time = \
        image_path_base + "Stance_Time_ignored_along_cycle.png"
    image_path_toe_mean_cycle = image_path_base + "Toe_mean_cycle.png"
    image_path_toe_max = image_path_base + "Toe_max_along_cycle.png"
    image_path_stride = image_path_base + "Stride_along_cycle.png"

    chart_style = TableStyle(
        [
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ("VALIGN", (0, 0), (-1, -1), 'CENTER'),
            ("LEFTPADDING", (0, 0), (-1, -1), 1),
            ("RIGHTPADDING", (0, 0), (-1, -1), 1),

        ]
    )
    image_height = 205
    image_width = 260
    image_grf_mean_cycle = Image(image_path_grf_mean_cycle,
                                 height=image_height, width=image_width)
    image_grf_impulse = Image(image_path_grf_impulse,
                              height=image_height, width=image_width)
    image_grf_max = Image(image_path_grf_max,
                          height=image_height, width=image_width)
    image_stance_time = Image(image_path_stance_time,
                              height=image_height, width=image_width)
    # image_toe_mean_cycle = Image(image_path_toe_mean_cycle,
    #                              height=image_height, width=image_width)
    # image_toe_max = Image(image_path_toe_max,
    #                       height=image_height, width=image_width)
    # image_stride = Image(image_path_stride,
    #                      height=image_height, width=image_width)

    chart_sole_sensor = Table([[image_grf_mean_cycle, image_stance_time],
                               [image_grf_max, image_grf_impulse]])
    # chart_kinematics = Table([[image_toe_mean_cycle, image_toe_max],
    #                           [image_stride, None]])

    spacer_10_point = Spacer(1, 10)
    spacer_30_point = Spacer(1, 30)
    spacer_200_point = Spacer(1, 200)

    story.append(paragraph_title)
    story.append(spacer_10_point)
    story.append(paragraph_sub1_title)
    story.append(spacer_10_point)
    story.append(table_sub1)
    story.append(spacer_30_point)
    story.append(paragraph_sub2_title)
    story.append(spacer_10_point)
    story.append(table_sub2)
    story.append(spacer_200_point)
    story.append(paragraph_graph_title)
    story.append(spacer_10_point)
    story.append(paragraph_graph_sole_sensor)
    story.append(spacer_10_point)
    story.append(chart_sole_sensor)
    story.append(spacer_10_point)
    story.append(paragraph_graph_kinematics)
    story.append(spacer_10_point)
    # story.append(chart_kinematics)

    doc.build(story)
