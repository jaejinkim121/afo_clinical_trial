import pandas as pd


left_path = "C:\\Users\\김재진\\source\\repos\\afo_clinical_trial\\bag\\log_2024-04-27-15-07-59\\afo_gui-kinematics_y.csv"
left_data = pd.read_csv(left_path)
left_time = left_data["Time"]
start_time = left_time[0]
left_time -= start_time

left_y = left_data["data_0"]
# print(left_y)
right_y = left_data["data_7"]
left_y = pd.DataFrame({"time":left_time, "value":left_y})
print(left_y)