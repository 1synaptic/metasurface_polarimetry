filname='C:\Users\User\Desktop\Polarimeter Project\metasurface_polarimetry\acquisition\data\small_metasurfaces\angles\top3_left3\0dgr\order_m2';
mkdir(filname);
cd(filname);


input('Starting repeatability measurement, makes sure that TXP_Server is started and press return to start measurement.');

    system('start C:\Users\User\Desktop\"Polarimeter Project"\metasurface_polarimetry\acquisition\TXP_PAX.exe');
    %cd 'C:\Users\User\Desktop\"Polarimeter Project"\metasurface_polarimetry\acquisition\data\calibration3'
    disp('Waiting for polarimeter to warm up');
    %pause(1*60)


        
        pause(5)
        
        %Single measurement version
        %ret=1;
        %ret = system('..\..\TXP_PAX.exe');
        %while ret ~= 0
        %   input('Error occured, press enter to restart current datapoint'); 
        %   ret = system('start ..\..\TXP_PAX_continuous.exe');
        %end
 
    system('taskkill /F /IM TXP_PAX.exe');


disp('DONE')
cd ..\..\..\..\..\..\