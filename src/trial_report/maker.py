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
import document


class ReportMaker:
    def __init__(self):
        self._root = tk.Tk()

        self._font_title = font.Font(family="Calibri", size=30, weight="bold")
        self._font_title_sub = font.Font(family="Calibri", size=15)
        self._font_sub_title = font.Font(family="Calibri", size=15, weight="bold")
        self._font_sub_sub = font.Font(family="Calibri", size=12)

        self._path_bag_file = "C:/"
        self._path_calibration_model_directory = "C:/"
        self._path_grf_model_file = "C:/"
        self._path_output_file = "C:/"
        self._shoe_size_list = [
            255, 260, 265, 270, 275, 280, 285
        ]
        outline_label = "flat"
        outline_button = "raised"
        outline_entry = "sunken"
        outline_option = "raised"

        self._label_title_main = tk.Label(
            self._root,
            text="AFO Report Maker",
            font=self._font_title,
            height=1,
            width=30,
            relief=outline_label
        )

        self._label_title_sub = tk.Label(
            self._root,
            text="made by Jaejin Kim",
            font=self._font_title_sub,
            height=2,
            anchor='n',
            relief=outline_label
        )

        # Section 1 - select paretic side with radiobutton
        self._label_sub1 = tk.Label(
            self._root,
            text="1. Paretic side",
            font=self._font_sub_title,
            relief=outline_label
        )

        self._var_paretic_side = tk.IntVar(value=1)
        self._radiobutton_sub1_left = tk.Radiobutton(
            self._root,
            text="Left",
            font=self._font_title_sub,
            value=1,
            variable=self._var_paretic_side,
            relief=outline_label
        )
        self._radiobutton_sub1_right = tk.Radiobutton(
            self._root,
            text="Right",
            font=self._font_title_sub,
            value=2,
            variable=self._var_paretic_side,
            relief=outline_label
        )

        # Section 2 - Select bag file and update file path
        self._label_sub2 = tk.Label(
            self._root,
            text="2. Bag file",
            font=self._font_sub_title,
            relief=outline_label
        )

        self._button_sub2_search = tk.Button(
            self._root,
            width=10,
            text="Search",
            command=self.search_bag_file_path,
            relief=outline_button,
        )

        self._string_var_sub2_search = tk.StringVar()
        self._string_var_sub2_search.set(self._path_bag_file)

        self._entry_sub2_search = tk.Entry(
            self._root, width=60, state='readonly',
            textvariable=self._string_var_sub2_search,
            relief=outline_entry
        )

        # Section 3 - Patient info (Name, Body_weight, Shoe_size)
        self._label_sub3 = tk.Label(
            self._root,
            text="3. Patient Info",
            font=self._font_sub_title,
            relief=outline_label
        )

        self._label_sub3_name = tk.Label(
            self._root,
            text="Name",
            font=self._font_sub_sub,
            relief=outline_label
        )

        self._label_sub3_body_weight = tk.Label(
            self._root,
            text="Body Weight [kg]",
            font=self._font_sub_sub,
            relief=outline_label
        )

        self._label_sub3_shoe_size = tk.Label(
            self._root,
            text="Shoe size",
            font=self._font_sub_sub,
            relief=outline_label
        )

        self._string_var_sub3_name = tk.StringVar()
        self._string_var_sub3_body_weight = tk.StringVar()
        self._entry_sub3_name = tk.Entry(
            self._root, width=20,
            textvariable=self._string_var_sub3_name,
            relief=outline_entry
        )
        self._entry_sub3_body_weight = tk.Entry(
            self._root, width=20,
            textvariable=self._string_var_sub3_body_weight,
            relief=outline_entry
        )
        self._int_var_sub3_shoe_size = tk.IntVar()
        self._int_var_sub3_shoe_size.set(self._shoe_size_list[0])
        self._opt_sub3_shoe_size = tk.OptionMenu(
            self._root,
            self._int_var_sub3_shoe_size,
            *self._shoe_size_list
        )
        self._opt_sub3_shoe_size.config(width=15,
            relief=outline_option)

        self._label_sub4 = tk.Label(
            self._root,
            text="4. Walking Distance",
            font=self._font_sub_title,
            relief=outline_label
        )
        self._label_sub4_sub1 = tk.Label(
            self._root,
            text="unit: [m]",
            font=self._font_sub_sub,
            relief=outline_label
        )
        self._string_var_sub4_walking_distance = tk.StringVar()
        self._entry_sub4_walking_distance = tk.Entry(
            self._root, width=25,
            textvariable=self._string_var_sub4_walking_distance,
            relief=outline_entry
        )

        self._label_sub5 = tk.Label(
            self._root,
            text="5. Model Path",
            font=self._font_sub_title,
            relief=outline_label
        )
        self._label_sub5_calibration = tk.Label(
            self._root,
            text="Calibration Model Path",
            font=self._font_sub_sub,
            relief=outline_label
        )
        self._label_sub5_grf = tk.Label(
            self._root,
            text="GRF Model Path",
            font=self._font_sub_sub,
            relief=outline_label
        )

        self._string_var_sub5_calibraion_model_path = tk.StringVar()
        self._string_var_sub5_calibraion_model_path.set(
            self._path_calibration_model_directory
        )
        self._string_var_sub5_grf_model_path = tk.StringVar()
        self._string_var_sub5_grf_model_path.set(
            self._path_grf_model_file
        )
        self._entry_sub5_calibration = tk.Entry(
            self._root, width=30,
            textvariable=self._string_var_sub5_calibraion_model_path,
            state='readonly',
            relief=outline_entry
        )
        self._entry_sub5_grf = tk.Entry(
            self._root, width=30,
            textvariable=self._string_var_sub5_grf_model_path,
            state='readonly',
            relief=outline_entry
        )
        self._button_sub5_calibration_search = tk.Button(
            self._root,
            width=10,
            text="Search",
            command=self.search_calibration_model_path,
            relief=outline_button
        )
        self._button_sub5_grf_search = tk.Button(
            self._root,
            width=10,
            text="Search",
            command=self.search_grf_model_path,
            relief=outline_button
        )

        self._label_sub6 = tk.Label(
            self._root,
            text="6. Output Path",
            font=self._font_sub_title,
            relief=outline_label
        )
        self._string_var_sub6_output_path = tk.StringVar()
        self._string_var_sub6_output_path.set(self._path_output_file)
        self._entry_sub6_output_path = tk.Entry(
            self._root, width=60,
            textvariable=self._string_var_sub6_output_path,
            state='readonly',
            relief=outline_entry
        )
        self._button_sub6_output_path = tk.Button(
            self._root,
            width=10,
            text="Search",
            command=self.search_output_path,
            relief=outline_button
        )

        self._button_make_output = tk.Button(
            self._root,
            width=20,
            height=3,
            text="Make Report",
            font=self._font_sub_title,
            command=self.make_clinical_report,
            relief=outline_button
        )

        self._button_exit = tk.Button(
            self._root,
            width=20,
            height=3,
            text="Exit",
            font=self._font_sub_title,
            command=self.exit_program,
            relief=outline_button
        )

    def run_gui(self):
        self._root.title("AFO Clinical Reporter")
        self._root.geometry("600x800+100+100")
        self._root.resizable(False, False)

        self._label_title_main.grid(row=0, column=0, columnspan=6)
        self._label_title_sub.grid(row=1, column=0, columnspan=6)

        self._label_sub1.grid(row=2, column=0, columnspan=6,
                              sticky='w', padx='30', pady='5')
        self._radiobutton_sub1_left.grid(row=3, column=0, columnspan=2,
                                         pady='5')
        self._radiobutton_sub1_right.grid(row=3, column=2, columnspan=2,
                                          pady='5')

        self._label_sub2.grid(row=4, column=0, columnspan=6,
                              sticky='w', padx='30', pady='5')
        self._entry_sub2_search.grid(row=5, column=0, columnspan=4,
                                     sticky='e', padx='5', pady='5')
        self._button_sub2_search.grid(row=5, column=4,
                                      sticky='w', padx='5', pady='5')

        self._label_sub3.grid(row=6, column=0, columnspan=6,
                              sticky='w', padx='30', pady='5')
        self._label_sub3_name.grid(row=7, column=0, columnspan=2,
                                   sticky='w', padx='44', pady='5')
        self._entry_sub3_name.grid(row=7, column=2, columnspan=4,
                                   sticky='w')
        self._label_sub3_body_weight.grid(row=8, column=0, columnspan=2,
                                          sticky='w', padx='44', pady='5')
        self._entry_sub3_body_weight.grid(row=8, column=2, columnspan=4,
                                          sticky='w')
        self._label_sub3_shoe_size.grid(row=9, column=0, columnspan=2,
                                        sticky='w', padx='44', pady='5')
        self._opt_sub3_shoe_size.grid(row=9, column=2, columnspan=4,
                                      sticky='w')

        self._label_sub4.grid(row=10, column=0, columnspan=6,
                              sticky='w', padx='30', pady='5')
        self._entry_sub4_walking_distance.grid(
            row=11, column=0, columnspan=2,
            sticky='w', padx='44', pady='5'
        )
        self._label_sub4_sub1.grid(row=11, column=2, columnspan=2,
                                   sticky='w', padx='5', pady='5')

        self._label_sub5.grid(row=12, column=0, columnspan=6,
                              sticky='w', padx='30', pady='5')
        self._label_sub5_calibration.grid(row=13, column=0, columnspan=2,
                                          sticky='w', padx='44', pady='5')
        self._entry_sub5_calibration.grid(
            row=13, column=2, columnspan=2, sticky='w', padx='5')
        self._button_sub5_calibration_search.grid(
            row=13, column=4, sticky='w', padx='5', pady='5'
        )
        self._label_sub5_grf.grid(row=14, column=0, columnspan=2,
                                  sticky='w', padx='44', pady='5')
        self._entry_sub5_grf.grid(row=14, column=2, columnspan=2,
                                  sticky='w', padx='5')
        self._button_sub5_grf_search.grid(row=14, column=4,
                                          sticky='w', padx='5', pady='5')

        self._label_sub6.grid(row=15, column=0, columnspan=6,
                              sticky='w', padx='30', pady='5')
        self._entry_sub6_output_path.grid(
            row=16, column=0, columnspan=4,
            sticky='e', padx='5', pady='5')
        self._button_sub6_output_path.grid(
            row=16, column=4, columnspan=2,
            sticky='w', padx='5', pady='5')
        self._button_make_output.grid(row=17, column=0, columnspan=3,
                                      pady='15')
        self._button_exit.grid(row=17, column=2, columnspan=3, pady='15')
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
        self._path_grf_model_file = \
            filedialog.askopenfilename(
                initialdir="C:\\Users\\",
                title='Select GRF model file'
            )
        self._string_var_sub5_grf_model_path.set(
            self._path_grf_model_file
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
        save_path = self._path_output_file + 'report_230816_.pdf'
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

        calib_model_path = self._path_calibration_model_directory + '/'
        grf_model_path = self._path_grf_model_file

        body_weight = self._string_var_sub3_body_weight.get()  # 장비무게포함
        if self._var_paretic_side == 1:
            paretic_side = 'L'
        else:
            paretic_side = 'R'
        sole_size = self._int_var_sub3_shoe_size.get()

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

        # toe_clearance_data = \
        #     ClinicalIndexJJ.get_clinical_index_max_toe_clearance(
        #         left_toe_path, right_toe_path)
        toe_clearance_data = [1,1,1]

        # stride_data = ClinicalIndexJJ.get_clinical_index_gait_speed_imu(
        #     stride_path)
        gait_speed_imu_data = [1, 1, 1]
        gait_speed_distance_data = [1, 1, 1]

        grf_max_data, grf_impulse_data = \
            ClinicalIndexMH.get_symmetry_index_grf(
                start_time=start_time,
                left_path=left_sole_path,
                right_path=right_sole_path,
                paretic_path=paretic_gait_path,
                non_paretic_path=nonparetic_gait_path,
                model_path_calib=calib_model_path,
                model_path_grf=grf_model_path,
                # GRF raw data 저장 경로
                raw_data_save_path=\
                    '../../data/report/2023-08-16/session_name/',
                # cycle별 timeseries data 저장 경로
                cycle_timeseries_data_save_path=\
                    '../../graph/2023-08-16/session_name/',
                size=str(sole_size),
                paretic_side=paretic_side,
                body_weight=float(body_weight),
                ignore_cycle=(None, None)
                )

        stance_time_data = ClinicalIndexMH.get_symmetry_index_stanceTime(
            start_time=start_time,
            paretic_path=paretic_gait_path,
            non_paretic_path=nonparetic_gait_path,
            ignore_cycle=(None, None)
            )

        data_analysis = document.ClinicalAnalysis()
        data_analysis.grf_max = grf_max_data
        data_analysis.grf_impulse = grf_impulse_data
        data_analysis.toe_clearance = toe_clearance_data
        data_analysis.stance_time = stance_time_data
        data_analysis.gait_speed_imu = gait_speed_imu_data
        data_analysis.gait_speed_distance = gait_speed_distance_data
        data_analysis.subject_name = self._string_var_sub3_name.get()
        data_analysis.age = str(30)
        data_analysis.weight = body_weight
        # data_analysis.test_date = None
        # data_analysis.test_label = None
        # data_analysis.session_type = None
        if self._var_paretic_side == 1:
            data_analysis.paretic_side = "Left"
        else:
            data_analysis.paretic_side = "Right"
        data_analysis.sole_size = sole_size
        # data_analysis.sensor_calibration_date = None

        document.make_report(save_path, data_analysis)

    def exit_program(self):
        self._root.quit()
        self._root.destroy()
