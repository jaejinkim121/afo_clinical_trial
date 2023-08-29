import tkinter as tk
from tkinter import font, filedialog
from reportlab.pdfgen import canvas
from reportlab.pdfbase import *
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import *
from reportlab.lib.styles import *
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import os
from bagpy import bagreader
import pandas as pd
import numpy as np

from jj import ClinicalIndexJJ
from mh import ClinicalIndexMH


class ReportMaker:
    def __init__(self):
        self._root = tk.Tk()

        self._font_title = font.Font(family="Calibri", size=30, weight="bold")
        self._font_sub_title = font.Font(family="Calibri", size=15)

        self._path_bag_file = "C:/"
        self._path_calibration_model_directory = "C:/"
        self._path_grf_model_directory = "C:/"
        self._path_output_file = "C:/"
        self._shoe_size_list = [
            255, 260, 265, 270, 275, 280, 285
        ]

        self._label_title_main = tk.Label(
            self._root,
            text="AFO Report Maker",
            font=self._font_title,
            height=1,
            width=30,
        )

        self._label_title_sub = tk.Label(
            self._root,
            text="made by Jaejin Kim",
            font=self._font_sub_title,
            height=2,
            anchor='n'
        )

        # Section 1 - select paretic side with radiobutton
        self._label_sub1 = tk.Label(
            self._root,
            text="1. Paretic side",
            font=self._font_sub_title,
            padx='30',
        )

        self._var_paretic_side = tk.IntVar(value=1)
        self._radiobutton_sub1_left = tk.Radiobutton(
            self._root,
            text="Left",
            font=self._font_sub_title,
            value=1,
            variable=self._var_paretic_side,
            pady='10',
        )
        self._radiobutton_sub1_right = tk.Radiobutton(
            self._root,
            text="Right",
            font=self._font_sub_title,
            value=2,
            variable=self._var_paretic_side,
            pady='10'
        )

        # Section 2 - Select bag file and update file path
        self._label_sub2 = tk.Label(
            self._root,
            text="2. Bag file",
            font=self._font_sub_title,
            padx='30',
        )

        self._button_sub2_search = tk.Button(
            self._root,
            width=15,
            text="Search",
            command=self.search_bag_file_path
        )

        self._string_var_sub2_search = tk.StringVar()
        self._string_var_sub2_search.set(self._path_bag_file)

        self._entry_sub2_search = tk.Entry(
            self._root, width=60, state='readonly',
            textvariable=self._string_var_sub2_search
        )

        # Section 3 - Patient info (Name, Body_weight, Shoe_size)
        self._label_sub3 = tk.Label(
            self._root,
            text="3. Patient Info",
            font=self._font_sub_title,
            padx='30',
        )

        self._label_sub3_name = tk.Label(
            self._root,
            text="Name",
            font=self._font_sub_title,
            padx='30'
        )

        self._label_sub3_body_weight = tk.Label(
            self._root,
            text="Body Weight [kg]",
            font=self._font_sub_title,
            padx='30'
        )

        self._label_sub3_shoe_size = tk.Label(
            self._root,
            text="Shoe size",
            font=self._font_sub_title,
            padx='30'
        )

        self._string_var_sub3_name = tk.StringVar()
        self._string_var_sub3_body_weight = tk.StringVar()
        self._entry_sub3_name = tk.Entry(
            self._root, width=20,
            textvariable=self._string_var_sub3_name
        )
        self._entry_sub3_body_weight = tk.Entry(
            self._root, width=20,
            textvariable=self._string_var_sub3_body_weight
        )
        self._int_var_sub3_shoe_size = tk.IntVar()
        self._int_var_sub3_shoe_size.set(self._shoe_size_list[0])
        self._opt_sub3_shoe_size = tk.OptionMenu(
            self._root,
            self._int_var_sub3_shoe_size,
            *self._shoe_size_list,
        )
        self._opt_sub3_shoe_size.config(width=15)

        self._label_sub4 = tk.Label(
            self._root,
            text="4. Walking Distance",
            font=self._font_sub_title,
            padx='30',
        )
        self._label_sub4_sub1 = tk.Label(
            self._root,
            text="unit: [m]",
            font=self._font_sub_title,
            padx='15',
        )
        self._string_var_sub4_walking_distance = tk.StringVar()
        self._entry_sub4_walking_distance = tk.Entry(
            self._root, width=20,
            textvariable=self._string_var_sub4_walking_distance
        )

        self._label_sub5 = tk.Label(
            self._root,
            text="5. Model Path",
            font=self._font_sub_title,
            padx='30',
        )
        self._label_sub5_calibration = tk.Label(
            self._root,
            text="Calibration Model Path",
            font=self._font_sub_title,
        )
        self._label_sub5_grf = tk.Label(
            self._root,
            text="GRF Model Path",
            font=self._font_sub_title
        )

        self._string_var_sub5_calibraion_model_path = tk.StringVar()
        self._string_var_sub5_calibraion_model_path.set(
            self._path_calibration_model_directory
        )
        self._string_var_sub5_grf_model_path = tk.StringVar()
        self._string_var_sub5_grf_model_path.set(
            self._path_grf_model_directory
        )
        self._entry_sub5_calibration = tk.Entry(
            self._root, width=35,
            textvariable=self._string_var_sub5_calibraion_model_path,
            state='readonly'
        )
        self._entry_sub5_grf = tk.Entry(
            self._root, width=35,
            textvariable=self._string_var_sub5_grf_model_path,
            state='readonly'
        )
        self._button_sub5_calibration_search = tk.Button(
            self._root,
            width=10,
            text="Search",
            command=self.search_calibration_model_path
        )
        self._button_sub5_grf_search = tk.Button(
            self._root,
            width=10,
            text="Search",
            command=self.search_grf_model_path
        )

        self._label_sub6 = tk.Label(
            self._root,
            text="6. Output Path",
            font=self._font_sub_title,
            padx='30',
        )
        self._string_var_sub6_output_path = tk.StringVar()
        self._string_var_sub6_output_path.set(self._path_output_file)
        self._entry_sub6_output_path = tk.Entry(
            self._root, width=60,
            textvariable=self._string_var_sub6_output_path,
            state='readonly'
        )
        self._button_sub6_output_path = tk.Button(
            self._root,
            width=10,
            text="Search",
            command=self.search_output_path
        )

        self._button_make_output = tk.Button(
            self._root,
            width=20,
            height=3,
            text="Make Report",
            font=self._font_sub_title,
            command=self.make_clinical_report
        )

        self._button_exit = tk.Button(
            self._root,
            width=20,
            height=3,
            text="Exit",
            font=self._font_sub_title,
            command=self.exit_program
        )


    def run_gui(self):
        self._root.title("AFO Clinical Reporter")
        self._root.geometry("600x800+100+100")
        self._root.resizable(False, False)

        self._label_title_main.grid(row=0, column=0, columnspan=6)
        self._label_title_sub.grid(row=1, column=0, columnspan=6)

        self._label_sub1.grid(row=2, column=0, columnspan=6, sticky='w')
        self._radiobutton_sub1_left.grid(row=3, column=0, columnspan=2)
        self._radiobutton_sub1_right.grid(row=3, column=2, columnspan=2)

        self._label_sub2.grid(row=4, column=0, columnspan=6, sticky='w')
        self._entry_sub2_search.grid(row=5, column=0, columnspan=4, padx='30')
        self._button_sub2_search.grid(row=5, column=4, columnspan=2,
                                      sticky='w', padx='7')

        self._label_sub3.grid(row=6, column=0, columnspan=6, sticky='w', pady='10')
        self._label_sub3_name.grid(row=7, column=0, columnspan=2)
        self._entry_sub3_name.grid(row=7, column=2, columnspan=4,
                                   padx='15', pady='10', sticky='w')
        self._label_sub3_body_weight.grid(row=8, column=0, columnspan=2)
        self._entry_sub3_body_weight.grid(row=8, column=2, columnspan=4,
                                          padx='15', pady='10', sticky='w')
        self._label_sub3_shoe_size.grid(row=9, column=0, columnspan=2)
        self._opt_sub3_shoe_size.grid(row=9, column=2, columnspan=4,
                                      padx='15', pady='10', sticky='w')

        self._label_sub4.grid(row=10, column=0, columnspan=6, sticky='w')
        self._entry_sub4_walking_distance.grid(row=11, column=0, columnspan=2)
        self._label_sub4_sub1.grid(row=11, column=2, columnspan=2, sticky='w')

        self._label_sub5.grid(row=12, column=0, columnspan=6, sticky='w')
        self._label_sub5_calibration.grid(row=13, column=0, columnspan=2)
        self._entry_sub5_calibration.grid(
            row=13, column=2, columnspan=2, sticky='w')
        self._button_sub5_calibration_search.grid(row=13, column=4)
        self._label_sub5_grf.grid(row=14, column=0, columnspan=2)
        self._entry_sub5_grf.grid(row=14, column=2, columnspan=2, sticky='w')
        self._button_sub5_grf_search.grid(row=14, column=4)

        self._label_sub6.grid(row=15, column=0, columnspan=6, sticky='w')
        self._entry_sub6_output_path.grid(
            row=16, column=0, columnspan=4, sticky='w', padx='30')
        self._button_sub6_output_path.grid(
            row=16, column=4, columnspan=2, sticky='w')
        self._button_make_output.grid(row=17, column=0, columnspan=2, pady='30')
        self._button_exit.grid(row=17, column=3, columnspan=2)
        #    label_sub3.grid(row=6, column=0, columnspan=6, sticky='w')
        # label_sub4.grid(row=6, column=0)
        # label_sub5.grid(row=7, column=0)
        # label_sub6.grid(row=8, column=0)
        # label_sub4_sub1.grid(row=9, column=0)

        self._root.mainloop()

    def search_bag_file_path(self):
        self._path_bag_file = filedialog.askopenfilename(
            initialdir='C:\\Users\\',
            title='Select Bag File',)
        self._string_var_sub2_search.set(self._path_bag_file)
        ...

    def search_calibration_model_path(self):
        self._path_calibration_model_directory = \
            filedialog.askdirectory(
                initialdir="C:\\Users\\",
                title='Select calibration model directory'
            )
        self._string_var_sub5_calibraion_model_path.set(
            self._path_calibration_model_directory
        )

    def search_grf_model_path(self):
        self._path_grf_model_directory = \
            filedialog.askdirectory(
                initialdir="C:\\Users\\",
                title='Select GRF model directory'
            )
        self._string_var_sub5_grf_model_path.set(
            self._path_grf_model_directory
        )

    def search_output_path(self):
        self._path_output_file = \
            filedialog.askdirectory(
                initialdir="C:\\Users\\",
                title='Select output file directory'
            )
        self._string_var_sub6_output_path.set(
            self._path_output_file
        )

    def make_clinical_report(self):
        # Read bag file
        path = self._path_bag_file
        save_path = self._path_output_file + 'report_df_230816_.csv'
        bag = bagreader(path)
        start_time = bag.start_time

        # To filter specific topics with interests
        TOPIC_MH = (
            "/afo_sensor/soleSensor_left",
            "/afo_sensor/soleSensor_right",
            "/afo_detector/gait_paretic",
            "/afo_detector/gait_nonparetic"
        )

        TOPIC_JJ = ("/afo_gui/left_toe_clearance",
                    "/afo_gui/right_toe_clearance",
                    "/afo_gui/stride")

        # Definition of Clinical Indices
        # Sample
        cadence_trial_mean = 0.0
        variables = [("vGRF", 1.0, "[N]"), ("Gait Speed", 21.0, "[m/s]")]

        left_toe_path = ""
        right_toe_path = ""
        stride_path = ""

        left_sole_path = ""
        right_sole_path = ""
        paretic_gait_path = ""
        nonparetic_gait_path = ""

        calib_model_path = '../../model/CHAR_230815_280_LP/'
        GRF_model_path = '../../model/GRF_230815/LSTM_GRF.pt'

        BW = float(85.1)  # 장비무게포함
        paretic_side = 'L'
        size = '280'

        # Read Topics and calculate clinical index
        for topic in bag.topics:
            msg_topic = bag.message_by_topic(topic)

            # Use own module and methods
            if topic == TOPIC_MH[0]:
                left_sole_path = msg_topic
            elif topic == TOPIC_MH[1]:
                right_sole_path = msg_topic
            elif topic == TOPIC_MH[2]:
                paretic_gait_path = msg_topic
            elif topic == TOPIC_MH[3]:
                nonparetic_gait_path = msg_topic

            if topic == TOPIC_JJ[0]:
                left_toe_path = msg_topic
            elif topic == TOPIC_JJ[1]:
                right_toe_path = msg_topic
            elif topic == TOPIC_JJ[2]:
                stride_path = msg_topic

        report_df = pd.DataFrame(columns=['mean_paretic', 'std_paretic',
                                          'mean_nonparetic', 'std_nonparetic',
                                          'symmetry'],
                                 index=['toeClearance', 'stride', 'GRFmax'
                                                                  'GRFimpulse',
                                        'stanceTime'])
        toe_clearance_data = \
            ClinicalIndexJJ.get_clinical_index_max_toe_clearance(
                left_toe_path, right_toe_path)

        stride_data = ClinicalIndexJJ.get_clinical_index_gait_speed_imu(
            stride_path)

        GRF_maximum_data = \
            ClinicalIndexMH.get_symmetry_index_GRFmax(start_time=start_time,
                                                      leftPath=left_sole_path,
                                                      rightPath=right_sole_path,
                                                      pareticPath=paretic_gait_path,
                                                      nonpareticPath=nonparetic_gait_path,
                                                      modelPathCalib=calib_model_path,
                                                      modelPathGRF=GRF_model_path,
                                                      size=size,
                                                      paretic_side=paretic_side,
                                                      BW=BW)

        GRF_impulse_data = \
            ClinicalIndexMH.get_symmetry_index_GRFimpulse(
                start_time=start_time,
                leftPath=left_sole_path,
                rightPath=right_sole_path,
                pareticPath=paretic_gait_path,
                nonpareticPath=nonparetic_gait_path,
                modelPathCalib=calib_model_path,
                modelPathGRF=GRF_model_path,
                size=size,
                paretic_side=paretic_side,
                BW=BW)

        stance_time_data = ClinicalIndexMH.get_symmetry_index_stanceTime(
            start_time=start_time,
            pareticPath=paretic_gait_path,
            nonpareticPath=nonparetic_gait_path,
            paretic_side='L')

        # add report df
        # report_df.loc['toeClearance', :] = toe_clearance_data
        # report_df.loc['stride', :] = stride_data
        report_df.loc['GRFmax', :] = GRF_maximum_data
        report_df.loc['GRFimpulse', :] = GRF_impulse_data
        report_df.loc['stanceTime', :] = stance_time_data

        report_df.to_csv(
            save_path,
            sep=',',
            columns=['mean_paretic', 'std_paretic',
                     'mean_nonparetic', 'std_nonparetic',
                     'symmetry'],
            index=['toeClearance', 'stride', 'GRFmax',
                   'GRFimpulse', 'stanceTime']
        )

        # Document Formatting
        story = []

        doc = SimpleDocTemplate(
            "test.pdf",
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18)
        styles = getSampleStyleSheet()

        story.append(
            Paragraph(
                "Ankle Foot Orthosis - Clinical Trial Report",
                styles['Title']))
        data = []
        for i in range(10):
            data.append([])
        data[0].append("Subject Name: ")
        data[0].append("KJJ")
        data[1].append("Subject ID: ")
        data[1].append("RH-23-01")
        data[2].append()

        story.append(

            Table(data)
        )
        doc.build(story)
        ...

    def exit_program(self):
        self._root.quit()
        self._root.destroy()
