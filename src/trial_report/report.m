close all
clear all

col1 = [1,48,71]/255; col2 = [34,158,188]/255; col3 = [143, 203, 229]/255; col4 = [253,182,20]/255;  col5 = [246,133,31]/255;

%% Rosbag Reading
bag = rosbag('D:\OneDrive - SNU\Projects\afo_clinical_trial\bag\log_2023-08-16-15-56-59.bag');
%sync = select(bag, 'Topic', '/afo_gui/sync');
soleleft = select(bag, 'Topic', '/afo_sensor/soleSensor_left');
soleright = select(bag, 'Topic', '/afo_sensor/soleSensor_right');
plantar = select(bag, 'Topic', '/afo_controller/motor_data_plantar');
dorsi = select(bag, 'Topic', '/afo_controller/motor_data_dorsi');
gait_paretic = select(bag, 'Topic', '/afo_detector/gait_paretic');
gait_non_paretic = select(bag, 'Topic', '/afo_detector/gait_nonparetic');
pf_run = select(bag, 'Topic', 'afo_gui/plantar_run');
%dorsi_neutral_position = select(bag, 'Topic', '/afo_controller/dorsiNeutralPosition');
rosout = select(bag, 'Topic', '/rosout');
initialTime = rosout.MessageList.Time(1);
% 
% sync = msg_to_array(sync, initialTime);
soleleft = msg_to_array(soleleft, initialTime);
soleright = msg_to_array(soleright, initialTime);
plantar = msg_to_array(plantar, initialTime);
dorsi = msg_to_array(dorsi, initialTime);
gait_paretic = msg_to_array(gait_paretic, initialTime);
gait_non_paretic = msg_to_array(gait_non_paretic, initialTime);
pf_run = msg_to_array(pf_run, initialTime);
%dnp = msg_to_array(dorsi_neutral_position, initialTime);

%% Main Body

%% Current Tracking
% s = 0;
% t = 0;
% for i=10000:30000
%     s = s + abs(plantar(i,6) * (plantar(i+1,1)-plantar(i,1)));
% end
% 
% figure;
% hold on;
% xlimit = [plantar(10000,1) plantar(30000,1)];
% ylimit = [-14 1];
% plot(plantar(:,1), plantar(:,6));
% xlim(xlimit);
% ylim(ylimit);
% title("Plantar Motor Current", 'FontSize', 28);
% xlabel('time [s]', 'FontSize', 28);
% ylabel('Current [A]', 'FontSize', 28);
% ax = gca;
% ax.FontSize = 28;
% dim = [0.5, 0.8, 0.1, 0.1];
% str = "Integrated Current = " + round(s, 2) + " [A\cdots], " + "Total time = " + round((xlimit(2) - xlimit(1)),2) + " s";
% annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', 'FontSize', 18);
% 
% s = 0;
% t = 0;
% for i=10000:30000
%     s = s + abs(dorsi(i,6) * (dorsi(i+1,1)-dorsi(i,1)));
% end
% 
% figure;
% hold on;
% xlimit = [dorsi(10000,1) dorsi(30000,1)];
% %ylimit = [-14 1];
% plot(dorsi(:,1), dorsi(:,6));
% xlim(xlimit);
% %ylim(ylimit);
% title("Dorsi Motor Current", 'FontSize', 28);
% xlabel('time [s]', 'FontSize', 28);
% ylabel('Current [A]', 'FontSize', 28);
% ax = gca;
% ax.FontSize = 28;
% dim = [0.5, 0.8, 0.1, 0.1];
% str = "Integrated Current = " + round(s, 2) + " [A\cdots], " + "Total time = " + round((xlimit(2) - xlimit(1)),2) + " s";
% annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', 'FontSize', 18);
%% Plotting - Poster 전용
num_trial = fix(length(pf_run)/2);
trial_start = zeros(num_trial,1);
trial_end = zeros(length(pf_run) - num_trial,1);

for trial_num=1:num_trial
    trial_start(trial_num) = pf_run(2 * trial_num - 1, 1);
    trial_end(trial_num) = pf_run(2 * trial_num, 1);
end
for trial_num = 1:num_trial
    init_time = trial_start(trial_num);
    final_time = trial_end(trial_num);
    duration = final_time - init_time;
    
    xlimit = [init_time final_time];

    fig = figure();
    subplot(3,1,1);
    hold on;
    plot(soleleft(:,1), soleleft(:,2), '-', 'color', col1, 'LineWidth', 2);
    plot(soleleft(:,1), soleleft(:,3), '--', 'color', col1, 'LineWidth', 2);
    plot(soleleft(:,1), soleleft(:,4), '-.', 'color', col1, 'LineWidth', 2);
    plot(soleleft(:,1), soleleft(:,5), "-", 'color', col2, 'LineWidth', 2);
    plot(soleleft(:,1), soleleft(:,6), '--', 'color', col2, 'LineWidth', 2);
    plot(soleleft(:,1), soleleft(:,7), "-.", 'color', col2, 'LineWidth', 2);
    for i=1:length(gait_paretic(:,1))
        if(gait_paretic(i,1) < init_time)
            continue;
        elseif (gait_paretic(i,1) > final_time)
            continue;
        end
        if(gait_paretic(i, 2) == 1)
            xline(gait_paretic(i, 1), 'color', col5, 'LineWidth', 2);
        else
            xline(gait_paretic(i, 1), 'color', col3, 'LineWidth', 2);
        end
    end

    xlim(xlimit);
    title("Sole Sensor Value (Paretic Side)", 'FontSize', 28);
    xlabel('time [s]', 'FontSize', 28);
    ylabel('FSR Voltage [V]', 'FontSize', 28);
    ax = gca;
    ax.FontSize = 28;

    subplot(3,1,2);
    hold on;
    plot(plantar(:,1), -plantar(:,4), 'LineWidth', 1.5);
    plot(plantar(:,1), -plantar(:,7), 'LineWidth', 1.5);
    for i=1:length(gait_paretic(:,1))
        if(gait_paretic(i,1) < init_time)
            continue;
        elseif (gait_paretic(i,1) > final_time)
            continue;
        end
        if(gait_paretic(i, 2) == 1)
            xline(gait_paretic(i, 1), 'color', col5, 'LineWidth', 2);
        else
            xline(gait_paretic(i, 1), 'color', col3, 'LineWidth', 2);
        end
    end
    xlim(xlimit);
    xlabel('time[s]', 'FontSize', 28);
    ylabel('torque', 'FontSize', 28);
    title('Input Torque Profile', 'FontSize', 28);
    ax = gca;
    ax.FontSize = 28;
    %plot(gait_paretic(:,1), gait_paretic(:,2) * 30 + 0, 'k');
    %yyaxis right;
    subplot(3,1,3);
    hold on;
    plot(dorsi(:,1), dorsi(:,4), 'LineWidth', 2);
    plot(dorsi(:,1), dorsi(:,7), 'LineWidth', 2);
    for i=1:length(gait_paretic(:,1))
        if(gait_paretic(i,1) < init_time)
            continue;
        elseif (gait_paretic(i,1) > final_time)
            continue;
        end
        if(gait_paretic(i, 2) == 1)
            xline(gait_paretic(i, 1), 'color', col5, 'LineWidth', 2);
        else
            xline(gait_paretic(i, 1), 'color', col3, 'LineWidth', 2);
        end
    end
    xlim(xlimit);
    xlabel('time[s]', 'FontSize', 28);
    ylabel('torque', 'FontSize', 28);
    title('Input Torque Profile', 'FontSize', 28);
    ax = gca;
    ax.FontSize = 28;
    % plot(dorsi(:,1), dorsi(:,7), 'g-');
    
end


% 
% figure;
% hold on;
% grid on;
% %for i=2:7
% %    plot(soleright(:,1), soleright(:,i));
% %end
% plot(soleright(:,1), soleright(:,2));
% plot(soleright(:,1), soleright(:,6));
% %legend('1','2','3','4','5','6');
% 
% figure;
% hold on;
% grid on;
% plot(plantar(:,1), plantar(:,7),'k');
% plot(plantar(:,1), plantar(:,4), 'r');


%% Function

function f = msg_to_array(msg, initialTime)
    t= msg.MessageList.Time - initialTime;
    data = readMessages(msg, 'DataFormat', 'struct');
    r_data = zeros(length(data), length(data{1}.Data));
    for i=1:length(data)
        r_data(i,:) = data{i}.Data;
    end
    f = [t' ; r_data']';
end