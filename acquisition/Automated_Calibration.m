%% Specify values in absolute coordinates for polarization optics.
clear;
pol_hor = 18.9723;
qwp_at_rcp = 12.75;
qwp_at_lcp = mod(qwp_at_rcp + 90, 360);
last_polarizer_at_45 = 105;
serial_rotation_mount = 55000517; % serial number of 1st motor 
serial_rotation_mount2 = 55000631; % serial number of rotation stage 
serial_rotation_stage = 83839448; % serial number of 2nd rotation mount
% for reference: the polarizer after the polarimeter is vertical at 140 degrees on the
% mount

global limit_in_beam
global limit_out_beam

limit_in_beam = 0; % point at which power meter is in the beam
limit_out_beam = 0.649984;% point at which the power meter is out of the beam

%% Initialize serial communications with motors, the power meter, and the daq

addpath('..');
fpos    = get(0,'DefaultFigurePosition'); % figure default position
fpos(3) = 640; % figure window size;Width
fpos(4) = 480; % Heights
f = figure('Position', fpos,...
           'Menu','None',...
           'Name','APT GUI');
       
global h_rot_mount
global h_rot_stage
global h_rot_mount2
       
h_rot_mount = actxcontrol('MGMOTOR.MGMotorCtrl.1',[20 20 600 480], f);
h_rot_stage = actxcontrol('MGMOTOR.MGMotorCtrl.1',[600 20 600 480], f);
h_rot_mount2 = actxcontrol('MGMOTOR.MGMotorCtrl.1',[20 480 600 480], f);
h_rot_mount.StartCtrl;
h_rot_stage.StartCtrl;
h_rot_mount2.StartCtrl;

set(h_rot_mount,'HWSerialNum', serial_rotation_mount);
set(h_rot_stage,'HWSerialNum', serial_rotation_stage);
set(h_rot_mount2,'HWSerialNum', serial_rotation_mount2);

h_rot_mount.Identify;
h_rot_stage.Identify;
h_rot_mount2.Identify;

global pwr_meter;

% Now configure ThorLabs power meter
pwr_meter = instrfind('Type', 'visa-usb');
if isempty(pwr_meter)
    pwr_meter = visa('ni','USB0::0x1313::0x8078::P0004265::INSTR');
else
    fclose(pwr_meter);
    pwr_meter = pwr_meter(1);
end
    
fopen(pwr_meter);
disp(query(pwr_meter, '*IDN?'));
fprintf(pwr_meter, 'SENS:CORR:WAV 532');

% configure the daq unit
global adc
adc = daq.createSession('ni');
% add channels to the daq
ch1=addAnalogInputChannel(adc,'Dev1','ai0','Voltage');
ch2=addAnalogInputChannel(adc,'Dev1','ai1','Voltage');
ch3=addAnalogInputChannel(adc,'Dev1','ai2','Voltage');
ch4=addAnalogInputChannel(adc,'Dev1','ai3','Voltage');
ch1.TerminalConfig = 'SingleEnded';
ch2.TerminalConfig = 'SingleEnded';
ch3.TerminalConfig = 'SingleEnded';
ch4.TerminalConfig = 'SingleEnded';

%dark current measurement
h_rot_stage.SetAbsMovePos(0, limit_in_beam);
h_rot_stage.MoveAbsolute(0,1);  
dark = daq_measure(5, 'data/dark.txt');

%% Carry out the linear part of the calibration

input('Assure that the linear polarizer (no qwp) is in place, and that the system is aligned. Press return to continue.');
linear_angles = 0:5:359; % range of angles at which to test

default_duration = 0.5; % measurement duration in seconds

figure % opens new figure window

cd 'data\polarizer_only'; % cd into a new directory for the linear polarization data
addpath('.');
figure;
xlabel('Linear polarizer angle');
ylabel('Power (a.u.)');
title('Linear polarization calibration data.');
xlim([0 max(linear_angles)]);
hold on %hold plot
for i = 1:length(linear_angles)
    curr_angle = linear_angles(i);
    h_rot_mount.SetAbsMovePos(0, pol_hor + curr_angle); % set a move to the angular offset from 0
    h_rot_mount.MoveAbsolute(0,0); % now move the polarizer
    
    pwr = check_beam_power(); % now get the power of the beam
    pwr = pwr/(0.001); % get the power in miliwatts
    file_name = [num2str(curr_angle), 'deg_',num2str(pwr),'.txt']; % file name
    dat = daq_measure(default_duration, file_name); % measure the voltage on the photodiodes
    dat = dat - dark;
    disp(['Completed linear polarizer measurement ', num2str(i), ' of ', num2str(length(linear_angles)), '.']);
    plot(curr_angle, dat(1)/pwr, 'bo'); %plotting results
    plot(curr_angle, dat(2)/pwr, 'ko');
    plot(curr_angle, dat(3)/pwr, 'go');
    plot(curr_angle, dat(4)/pwr, 'ro');
end
hold off

%% Move on to the QWP part of the calibration, RCP
cd '..';
input(['Place the QWP oriented at ', num2str(qwp_at_rcp), ' in front of the linear polarizer and press return to continue.']);
cd 'data\qwp_R'; % cd into a new directory for the linear polarization data
addpath('.');
default_duration = 0.5;
qwp_angles = 0:5:359;

figure;
xlabel('Absolute angle');
ylabel('Power (a.u.)');
title('RCP calibration data.');
xlim([0 max(qwp_angles)]);
hold on %hold plot
for i = 1:length(qwp_angles)
    curr_angle = qwp_angles(i);
    curr_abs_angle_pol = mod(pol_hor + curr_angle, 360);
    curr_abs_angle_qwp = mod(qwp_at_rcp + curr_angle, 360);
    h_rot_mount.SetAbsMovePos(0, curr_abs_angle_pol); % set a move to the angular offset from 0
    h_rot_mount.MoveAbsolute(0,0); % now move the polarizer
    h_rot_mount2.SetAbsMovePos(0, curr_abs_angle_qwp); % set a move to the angular offset from 0
    h_rot_mount2.MoveAbsolute(0,0); % now move the polarizer
    pwr = check_beam_power(); % now get the power of the beam
    pwr = pwr/0.001; % get the power in miliwatts
    
    file_name = ['p', num2str(curr_abs_angle_pol), 'deg_r', num2str(curr_abs_angle_qwp), 'deg_', num2str(pwr),'.txt']; % file name
    dat = daq_measure(default_duration, file_name); % measure the voltage on the photodiodes
    dat = dat - dark;
    plot(curr_angle, dat(1)/pwr, 'bo'); %plotting result
    plot(curr_angle, dat(2)/pwr, 'ko');
    plot(curr_angle, dat(3)/pwr, 'go');
    plot(curr_angle, dat(4)/pwr, 'ro');    
    disp(['Completed QWP1 measurement ', num2str(i), ' of ', num2str(length(qwp_angles)), '.']);
end
hold off

% Move on to the second QWP part of the calibration, LCP

default_duration = 0.5;
%input(['Place the QWP oriented at ', num2str(qwp_at_lcp), ' in front of the linear polarizer and press return to continue.']);
cd 'data\qwp_L'; % cd into a new directory for the linear polarization data
addpath('.');
addpath('..');
qwp_angles = 0:5:360;

figure;
xlabel('Absolute angle');
ylabel('Power (a.u.)');
title('LCP calibration data.');
xlim([0 max(qwp_angles)]);
hold on %hold plot

for i = 1:length(qwp_angles)
    curr_angle = qwp_angles(i);
    curr_abs_angle_pol = mod(pol_hor + curr_angle, 360);
    curr_abs_angle_qwp = mod(qwp_at_lcp + curr_angle, 360);
    h_rot_mount.SetAbsMovePos(0, curr_abs_angle_pol); % set a move to the angular offset from 0
    h_rot_mount.MoveAbsolute(0,0); % now move the polarizer
    h_rot_mount2.SetAbsMovePos(0, curr_abs_angle_qwp); % set a move to the angular offset from 0
    h_rot_mount2.MoveAbsolute(0,0); % now move the polarizer
    pwr = check_beam_power(); % now get the power of the beam
    pwr = pwr/0.001; % get the power in miliwatts
    file_name = ['p', num2str(curr_abs_angle_pol), 'deg_r', num2str(curr_abs_angle_qwp), 'deg_', num2str(pwr),'.txt']; % file name
    dat = daq_measure(default_duration, file_name); % measure the voltage on the photodiodes
    dat = dat - dark;
    plot(curr_angle, dat(1)/pwr, 'bo'); %plotting result
    plot(curr_angle, dat(2)/pwr, 'ko');
    plot(curr_angle, dat(3)/pwr, 'go');
    plot(curr_angle, dat(4)/pwr, 'ro');    
    disp(['Completed QWP2 measurement ', num2str(i), ' of ', num2str(length(qwp_angles)), '.']);
end

hold off
disp('Calibration completed.');

%% Partial polarization state measurement

default_duration = 0.5;
input('Press return to begin partial polarization measurement.');
cd 'data\partial_pol3'; % cd into a new directory for the linear polarization data
addpath('..');
addpath('..');
pol_angles = 0:5:359;
%pol_angles = pol_angles - pol_hor;

figure;
xlabel('First polarizer angle');
ylabel('Power (a.u.)');
%title('Partial polarization calibration data.');
%xlim([0 max(qwp_angles)]);
hold on %hold plot

for i = 1:length(pol_angles)
    curr_angle = pol_angles(i);
    curr_abs_angle_pol = mod(pol_hor + curr_angle, 360);
    h_rot_mount.SetAbsMovePos(0, curr_abs_angle_pol); % set a move to the angular offset from 0
    h_rot_mount.MoveAbsolute(0, 1); % now move the polarizer, wait until complete
    %pwr = check_beam_power(); % now get the power of the beam
    %pwr = pwr/0.001; % get the power in miliwatts
    file_name = ['lp_', num2str(curr_abs_angle_pol), 'deg.txt']; % file name
    dat = daq_measure(default_duration, file_name); % measure the voltage on the photodiodes
    dat = dat - dark;
    disp(['Completed partial polarization measurement ', num2str(i), ' of ', num2str(length(pol_angles)), '.']);
    %plot(curr_angle, dat(1)/pwr, 'bo'); %plotting result
    %plot(curr_angle, dat(2)/pwr, 'ko');
    %plot(curr_angle, dat(3)/pwr, 'go');
    %plot(curr_angle, dat(4)/pwr, 'ro');       
end
hold off 
disp(['Completed partial polarization measurement.']);

