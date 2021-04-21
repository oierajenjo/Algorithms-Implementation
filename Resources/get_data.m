close all
clear all
clc
%% Get Paths to files
base = 'CompressorData';
list = dir(fullfile(base, '**', '*.mat'));

numTests = numel(list);

for iFile = 1:numTests
    lFiles(iFile) = append(list(iFile).folder,"\",list(iFile).name);
end
    clear list base iFile

%% Get data from each Path
% data = struct("Test_Nr","","Faulty","","T0","","Tc","","Tsh","",...
%     "Vexp_real","","Tsup_real","","Tsuc_real","","Tret_real","",...
%     "Tdis_real","","Psuc_real","","Pdis_real","","Evap_Fan_Speed_real","",...
%     "Cpr_real","","Cond_fan_speed_real","");
% data(1,1:15) = ["Test_Nr","Faulty","T0","Tc","Tsh","Vexp_real","Tsup_real",...
%     "Tsuc_real","Tret_real","Tdis_real","Psuc_real","Pdis_real","Evap_Fan_Speed_real",...
%     "Cpr_real","Cond_fan_speed_real"];
n  = 1;
for iFile = 1:numTests
    sim_data = load(lFiles(iFile)).sim_data;
    n2 = n+sim_data.EndSample;
    data(n:n2,1) = sim_data.test_no;
    data(n:n+sim_data.FaultSample-2,2) = 0;
    data(n+sim_data.FaultSample-1:n2,2) = 1;
    data(n:n2,3) = sim_data.T0;
    data(n:n2,4) = sim_data.Tc;
    data(n:n2,5) = sim_data.Tsh;
    data(n:n2,6) = sim_data.Vexp_real;
    data(n:n2,7) = sim_data.Tsup_real;
    data(n:n2,8) = sim_data.Tsuc_real;
    data(n:n2,9) = sim_data.Tret_real;
    data(n:n2,10) = sim_data.Tdis_real;
    data(n:n2,11) = sim_data.Psuc_real;
    data(n:n2,12) = sim_data.Pdis_real;
    data(n:n2,13) = sim_data.EvapFanSpeed_real;
    data(n:n2,14) = sim_data.Cpr_real;
    data(n:n2,15) = sim_data.CondFanSpeed_real;
    n = n2+1;
end 
% sorted_data = sortrows(data);
end_data(1,1:15) = ["Test_nr","Faulty","T0","Tc","Tsh","Vexp_real","Tsup_real",...
    "Tsuc_real","Tret_real","Tdis_real","Psuc_real","Pdis_real","EvapFanSpeed_real",...
    "Cpr_real","CondFanSpeed_real"];
end_data(2:n,1:15) = data;

%% Write CSV file
writematrix(end_data,'allData.csv')