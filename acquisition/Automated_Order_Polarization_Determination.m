%% Establish serial communication with motors and power meter

clear;
foldername = 'pol_det';
mkdir(['C:\Users\User\Desktop\Polarimeter Project\metasurface_polarimetry\acquisition\data\', foldername]);
serial_rotation_mount = 55000517; % serial number of 1st rotation mount 
serial_rotation_mount2 = 55000631; % serial number of 2nd rotation mount 

addpath('..');
fpos    = get(0,'DefaultFigurePosition'); % figure default position
fpos(3) = 640; % figure window size;Width
fpos(4) = 480; % Heights
f = figure('Position', fpos,...
           'Menu','None',...
           'Name','APT GUI');
       
global h_rot_mount
global h_rot_mount2
       
h_rot_mount = actxcontrol('MGMOTOR.MGMotorCtrl.1',[20 20 600 480], f);
h_rot_mount2 = actxcontrol('MGMOTOR.MGMotorCtrl.1',[20 480 600 480], f);
h_rot_mount.StartCtrl;
h_rot_mount2.StartCtrl;

set(h_rot_mount,'HWSerialNum', serial_rotation_mount);
set(h_rot_mount2,'HWSerialNum', serial_rotation_mount2);

h_rot_mount.Identify;
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

%% Now do alignment of waveplate and polarizer relative to first polarizer

input('Make sure that the first polarizer is at 45 degrees relative to the table, according to the commercial polarimeter. Place the second polarizer in front of it, being sure to center, and press return.');


n_angles=18;
figure
hold on

for n=1:3
    if n==1
        angles=linspace(0,180,n_angles);
    elseif n==2
        angles=linspace(angles(min_angle_index)-10,angles(min_angle_index)+10,n_angles);
    elseif n==3
        angles=linspace(angles(min_angle_index)-3,angles(min_angle_index)+3,n_angles);
    end
    pwrs = zeros(n_angles,1);
    for i=1:length(angles)
        h_rot_mount.SetAbsMovePos(0, angles(i));
        h_rot_mount.MoveAbsolute(0,1);   
        pause(1)
        fprintf(pwr_meter, 'MEAS:POW?');
        pwrs(i) = str2double(fscanf(pwr_meter));
        pwrs(i) = pwrs(i)/(0.001);
        plot(angles(i), pwrs(i), 'bo'); 
        pause(0.5)  
    end

    min_angle_index = find(pwrs == min(pwrs(:)));
end

angles=transpose(angles);
p = polyfit(angles,pwrs,2);
plot(angles, p(1)*angles.*angles+p(2)*angles+p(3));


min_angle = -p(2)/(2*p(1)) % this is the angle at which the pol is at -45
hor_angle = min_angle+45; % this is the angle at which the pol is horizontal
h_rot_mount.SetAbsMovePos(0, min_angle);
h_rot_mount.MoveAbsolute(0,1);   

hold off

input('Determined relative orientation of second polarizer. Place the QWP in between the two polarizers, being sure to center the beam in the clear aperture, and press return.')

n_angles=18;
figure

hold on

for n=1:2
    if n==1
        angles=linspace(0,90,n_angles);
    elseif n==2
        angles=linspace(angles(max_angle_index)-10,angles(max_angle_index)+10,n_angles);
    end
    pwrs = zeros(n_angles,1);
    for i=1:length(angles)
        h_rot_mount2.SetAbsMovePos(0, angles(i));
        h_rot_mount2.MoveAbsolute(0,1);   
        pause(1)
        fprintf(pwr_meter, 'MEAS:POW?');
        pwrs(i) = str2double(fscanf(pwr_meter));
        pwrs(i) = pwrs(i)/(0.001);
        plot(angles(i), pwrs(i), 'bo'); 
        pause(0.5)  
    end

    max_angle_index = find(pwrs == max(pwrs(:)));
end

angles=transpose(angles);
p = polyfit(angles,pwrs,2);
plot(angles, p(1)*angles.*angles+p(2)*angles+p(3));

qwp_at_rcp = -p(2)/(2*p(1)) %Max reading on power meter with qwp between crossed polarizers
qwp_at_lcp = mod(qwp_at_rcp + 90, 360) %LCP and RCP are arbitrary and can be switched

h_rot_mount2.SetAbsMovePos(0, qwp_at_rcp);
h_rot_mount2.MoveAbsolute(0,1); 
hold off

disp('Determined relative orientation of all polarization optics.')

%% Now complete the polarization determination.

input('Place the metasurface in the beam, align it, and place just the linear polarizer in the order of interest. Press return to continue.');

foldername = 'big_metasurface';
mkdir(['C:\Users\User\Desktop\Polarimeter Project\metasurface_polarimetry\acquisition\data\', foldername]);
cd(['C:\Users\User\Desktop\Polarimeter Project\metasurface_polarimetry\acquisition\data\',foldername]);
save_dir = 'order_2';
mkdir(save_dir);
cd(save_dir);
linear_pol_extension = 'pol_only';

pol_angles = 0:45:135; % polarizer angle testing points
linear_out_powers = zeros(1, length(pol_angles)); % create an array to store the power data

% linear polarizer loop
for i = 1:length(pol_angles)
    curr_angle = pol_angles(i) + hor_angle; % convert to coordinate system of rotation mount
    h_rot_mount.SetAbsMovePos(0, curr_angle);
    h_rot_mount.MoveAbsolute(0, 1);
    pause(0.5)
    fprintf(pwr_meter, 'MEAS:POW?'); % query the power
    linear_out_powers(i) = str2double(fscanf(pwr_meter)) / 0.001; % get the power
    disp(['Completed linear point ', num2str(i), ' of ', num2str(length(pol_angles)), '.']);
end

polarizer_dataset = horzcat(pol_angles', linear_out_powers'); % combine the data
csvwrite(linear_pol_extension, polarizer_dataset); % now save the data

input('Linear polarization data acquisition completed and saved. Please place QWP in front of linear polarizer and press return to continue.');
qwp_R_extension = 'qwp_R';

qwp_angles = 0:5:360;
qwp_R_out_powers = zeros(1, length(qwp_angles));

% qwp_R loop
for i = 1:length(qwp_angles)
    curr_pol_angle = mod(qwp_angles(i) + hor_angle, 360); % convert to coordinate system of rotation mount
    curr_qwp_angle = mod(qwp_angles(i) + qwp_at_rcp + 45, 360);
    h_rot_mount.SetAbsMovePos(0, curr_pol_angle);
    h_rot_mount2.SetAbsMovePos(0, curr_qwp_angle);
    h_rot_mount.MoveAbsolute(0, 0);
    h_rot_mount2.MoveAbsolute(0, 1);
    tic;
    while and(toc<36, or(IsMoving(h_rot_mount)==1, IsMoving(h_rot_mount2)==1))
       pause(1) 
    end
    pause(0.5)
    fprintf(pwr_meter, 'MEAS:POW?'); % query the power
    qwp_R_out_powers(i) = str2double(fscanf(pwr_meter)) / 0.001; % get the power
    disp(['Completed qwp_R point ', num2str(i), ' of ', num2str(length(qwp_angles)), '.']);
end

qwp_R_dataset = horzcat(qwp_angles', qwp_R_out_powers');
csvwrite(qwp_R_extension, qwp_R_dataset);

qwp_L_extension = 'qwp_L';
qwp_L_out_powers = zeros(1, length(qwp_angles));

% qwp_L loop
for i = 1:length(qwp_angles)
    curr_pol_angle = mod(qwp_angles(i) + hor_angle, 360); % convert to coordinate system of rotation mount
    curr_qwp_angle = mod(qwp_angles(i) + qwp_at_lcp + 45, 360);
    h_rot_mount.SetAbsMovePos(0, curr_pol_angle);
    h_rot_mount2.SetAbsMovePos(0, curr_qwp_angle);
    h_rot_mount.MoveAbsolute(0, 0);
    h_rot_mount2.MoveAbsolute(0, 1);
    tic;
    while and(toc<36, or(IsMoving(h_rot_mount)==1, IsMoving(h_rot_mount2)==1))
       pause(1) 
    end
    pause(0.5)
    fprintf(pwr_meter, 'MEAS:POW?'); % query the power
    qwp_L_out_powers(i) = str2double(fscanf(pwr_meter)) / 0.001; % get the power
    disp(['Completed qwp_L point ', num2str(i), ' of ', num2str(length(qwp_angles)), '.']);
end

qwp_L_dataset = horzcat(qwp_angles', qwp_L_out_powers');
csvwrite(qwp_L_extension, qwp_L_dataset);

disp('Measurement completed. Move to another diffraction order and change the directory name appropriately.');

%% Quickly compute the Stokes vector of the given order

qwp_R_intensity = mean(qwp_R_out_powers);
qwp_L_intensity = mean(qwp_L_out_powers);
A = zeros(length(polarizer_dataset) + 2, 4);
I = zeros(length(polarizer_dataset) + 2, 1);

for i = 1:length(polarizer_dataset)
    angle = -polarizer_dataset(i, 1);
    stokes_vector = [[1, 0, 0, 0]; [0, cosd(2*angle), sind(2*angle), 0]; [0, -sind(2*angle), cosd(2*angle), 0]; [0, 0, 0, 1]] * [1; 1; 0; 0];
    A(i, :) = stokes_vector';
    I(i, 1) = polarizer_dataset(i, 2);
end

A(length(polarizer_dataset) + 1, :) = [1, 0, 0, 1];
A(length(polarizer_dataset) + 2, :) = [1, 0, 0, -1];
I(length(polarizer_dataset) + 1, 1) = qwp_R_intensity;
I(length(polarizer_dataset) + 2, 1) = qwp_L_intensity;

polarization_state = pinv(A) * I;
polarization_state = polarization_state/polarization_state(1);
dop = sqrt(polarization_state(2)^2 + polarization_state(3)^2 + polarization_state(4)^2)
