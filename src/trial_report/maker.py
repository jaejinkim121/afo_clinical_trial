import tkinter as tk
from tkinter import font, filedialog
from bagpy import bagreader

import dataclasses
from utils import create_folder
import numpy as np
import pandas as pd
import json
import document
from define import *
from analysis import ClinicalAnalysis


def load_bag(bag: str, data_json):
    for dict_ in [x for x in data_json if x["bag"] == bag]:
        dict_["paretic_side"] = Side(dict_["paretic_side"])
        dict_["session_type"] = Session(dict_["session_type"])
        bag_ = from_dict(dict_)
        return bag_
    return None


class ReportMaker:
    def __init__(self, path):
        self._root = tk.Tk()

        self._bag = None

        self._font_title = font.Font(family="Calibri", size=30, weight="bold")
        self._font_title_sub = font.Font(family="Calibri", size=15)
        self._font_sub_title = font.Font(family="Calibri", size=15, weight="bold")
        self._font_sub_sub = font.Font(family="Calibri", size=12)

        self._path_default = path
        self._path_bag_file = path + '/bag'
        self._path_calibration_model_directory = path + '/model/'
        self._path_grf_model_file = path + '/model/'
        self._path_output_file = path + '/report/'
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

        # Section 1 - Select bag file and update file path
        self._label_sub1 = tk.Label(
            self._root,
            text="1. Bag file",
            font=self._font_sub_title,
            relief=outline_label
        )

        self._button_sub1_search = tk.Button(
            self._root,
            width=10,
            text="Search",
            command=self.search_bag_file_path,
            relief=outline_button,
        )

        self._string_var_sub1_search = tk.StringVar()
        self._string_var_sub1_search.set(self._path_bag_file)

        self._entry_sub1_search = tk.Entry(
            self._root, width=60, state='readonly',
            textvariable=self._string_var_sub1_search,
            relief=outline_entry
        )

        # Section 2 - select paretic side with radiobutton
        self._label_sub2 = tk.Label(
            self._root,
            text="1. Paretic side",
            font=self._font_sub_title,
            relief=outline_label
        )

        self._var_paretic_side = tk.IntVar(value=1)
        self._radiobutton_sub2_left = tk.Radiobutton(
            self._root,
            text="Left",
            font=self._font_title_sub,
            value=1,
            variable=self._var_paretic_side,
            relief=outline_label
        )
        self._radiobutton_sub2_right = tk.Radiobutton(
            self._root,
            text="Right",
            font=self._font_title_sub,
            value=2,
            variable=self._var_paretic_side,
            relief=outline_label
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
        self._entry_sub1_search.grid(row=3, column=0, columnspan=4,
                                     sticky='e', padx='5', pady='5')
        self._button_sub1_search.grid(row=3, column=4,
                                      sticky='w', padx='5', pady='5')

        self._label_sub2.grid(row=4, column=0, columnspan=6,
                              sticky='w', padx='30', pady='5')
        self._radiobutton_sub2_left.grid(row=5, column=0, columnspan=2,
                                         pady='5')
        self._radiobutton_sub2_right.grid(row=5, column=2, columnspan=2,
                                          pady='5')


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
        file_name = filedialog.askopenfilename(
            initialdir=self._path_default + '/bag',
            title='Select Bag File',
            filetypes=(("Bag files", "*.bag"), ("all files", "*.*"))
        )
        if not file_name:
            return
        self._path_bag_file = file_name
        self._string_var_sub1_search.set(self._path_bag_file)

        with open(self._path_default + '/bag_info.json', 'r') as f:
            bag_ = load_bag(file_name.split('/')[-1], json.load(f))
        if not bag_:
            return

        if bag_.paretic_side == Side.LEFT:
            self._radiobutton_sub2_left.select()
        else:
            self._radiobutton_sub2_right.select()
        self._string_var_sub3_name.set(bag_.name)
        self._string_var_sub3_body_weight.set(bag_.body_weight)
        self._int_var_sub3_shoe_size.set(bag_.sole_size)
        self._string_var_sub4_walking_distance.set(bag_.distance)
        self._string_var_sub5_calibraion_model_path.set(
            self._path_default + '/model/' + bag_.model_cell)
        self._string_var_sub5_grf_model_path.set(
            self._path_default + '/model/' + bag_.model_grf)


    def search_calibration_model_path(self):
        folder_name = \
            filedialog.askdirectory(
                initialdir=self._path_default + '/model/',
                title='Select calibration model directory'
            )

        if not folder_name:
            return
        self._path_calibration_model_directory = folder_name
        self._string_var_sub5_calibraion_model_path.set(
            self._path_calibration_model_directory
        )

    def search_grf_model_path(self):
        file_name = \
            filedialog.askopenfilename(
                initialdir=self._path_default + '/model/',
                title='Select GRF model file'
            )
        if not file_name:
            return
        self._path_grf_model_file = file_name
        self._string_var_sub5_grf_model_path.set(
            self._path_grf_model_file
        )

    def search_output_path(self):
        folder_name = \
            filedialog.askdirectory(
                initialdir=self._path_default + '/report/',
                title='Select output file directory'
            )
        if not folder_name:
            return
        self._path_output_file = folder_name
        self._string_var_sub6_output_path.set(
            self._path_output_file
        )

    def update_bag_info_json(self):
        bag_name = self._path_bag_file.split('/')[-1]
        with open(self._path_default + '/bag_info.json', 'r') as f:
            data = json.load(f)
        new_data = load_bag(bag_name, data)

        if new_data is None:
            test_date = bag_name.split('_')[1].split('-')
            test_date = test_date[0][2:] + test_date[1] + test_date[2]
            test_label = 'RH-' + test_date[2:4] + '-01'
            model_cell = \
                self._string_var_sub5_calibraion_model_path.get().split('/')[-1]
            grf_path_ = self._string_var_sub5_grf_model_path.get().split('/')
            model_grf = grf_path_[-2] + '/' + grf_path_[-1]
            calib_date = model_cell.split('_')[1]
            if self._var_paretic_side.get() == 1:
                side = Side.LEFT
            else:
                side = Side.RIGHT

            session = Session.TWO_MIN_OFF_CUE_OFF

            new_data = \
                Bag(bag_name,
                    test_date,
                    test_label,
                    model_cell,
                    model_grf,
                    self._string_var_sub3_name.get(),
                    float(self._string_var_sub3_body_weight.get()),
                    self._int_var_sub3_shoe_size.get(),
                    side,
                    session,
                    calib_date,
                    float(self._string_var_sub4_walking_distance.get()),
                    float(self._string_var_sub4_walking_distance.get())
                    )

            data.append(dataclasses.asdict(new_data))

            with open(self._path_default + '/bag_info.json',
                      'w', encoding='utf-8') as f:
                json.dump(data, f, indent='\t')

        return new_data

    def make_clinical_report(self):
        metadata_ = self.update_bag_info_json()

        # Read bag file
        path = self._path_bag_file

        save_path = self._path_default + "/report/data/" + \
            metadata_.test_label + "/" + metadata_.session_type + "/"
        create_folder(save_path)

        bag_raw = bagreader(path)
        start_time = bag_raw.start_time

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

        # Read Topics and calculate clinical index
        for topic in bag_raw.topics:
            msg_topic = bag_raw.message_by_topic(topic)

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
        toe_clearance_data = [1, 1, 1]

        # stride_data = ClinicalIndexJJ.get_clinical_index_gait_speed_imu(
        #     stride_path)
        gait_speed_imu_data = [1, 1, 1]
        gait_speed_distance_data = [1, 1, 1]

        grf_max_data, grf_impulse_data, stance_time_data = \
            ClinicalAnalysis.data_analysis_grf(
                meta_data=metadata_,
                default_path=self._path_default,
                start_time=start_time,
                left_path=left_sole_path,
                right_path=right_sole_path,
                paretic_path=paretic_gait_path,
                non_paretic_path=nonparetic_gait_path,
                save_each_cycle_flag=True
                )

        # Save statistics.csv
        pd_statistics = pd.DataFrame(
            columns=["paretic_mean", "paretic_stdev",
                     "non_paretic_mean", "non_paretic_stdev",
                     "symmetry"],
            index=["grf_max", "grf_impulse", "stance_time"]
            )
        pd_statistics.loc["grf_max"] = grf_max_data
        pd_statistics.loc["grf_impulse"] = grf_impulse_data
        pd_statistics.loc["stance_time"] = stance_time_data
        pd_statistics.to_csv(
            save_path + 'statistics.csv', sep=",", header=True, index=True
            )

        data_analysis = ClinicalDataset(
            limb_length={"Femur": 1, "Tibia": 2, "Foot": 3, "Pelvis": 4},
            grf_max=grf_max_data,
            grf_impulse=grf_impulse_data,
            toe_clearance=toe_clearance_data,
            stance_time=stance_time_data,
            gait_speed_imu=gait_speed_imu_data,
            gait_speed_distance=gait_speed_distance_data,
            metadata=metadata_
        )

        document.make_report(self._path_default, data_analysis)
        print("Complete report making")

    def exit_program(self):
        self._root.quit()
        self._root.destroy()
