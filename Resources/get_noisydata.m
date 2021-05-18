close all
clear all
clc
%% Get Paths to files
curDir = pwd;
parDir = strfind(curDir,filesep);
parDir = curDir(1:parDir(end)-1);
base = [parDir '\data\NoisyCompressorData*'];
list = dir(fullfile(base, '**', '*.mat'));

numTests = numel(list);

for iFile = 1:numTests
    lFiles(iFile) = append(list(iFile).folder,"\",list(iFile).name);
end
% clear list base iFile

%% Get data from each Path
n = 1;
for iFile = 1:numTests
    sim_data = load(lFiles(iFile)).sim_data;
    
    sNF_begin = 4000;
    sNF_end = sim_data.FaultSample-100;
    sF_begin = 10000;
%     sF_begin = sim_data.FaultSample+1500;
    sF_end = sim_data.EndSample;
    
    nNF = n+sNF_end-sNF_begin;
    nF = nNF+1+sF_end-sF_begin;
    
    data(n:nF,1) = sim_data.test_no;
    
    data(n:nNF,2) = sim_data.T0(sNF_begin:sNF_end);
    data(nNF+1:nF,2) = sim_data.T0(sF_begin:sF_end);
    
    data(n:nNF,3) = sim_data.Tc(sNF_begin:sNF_end);
    data(nNF+1:nF,3) = sim_data.Tc(sF_begin:sF_end);
    
    data(n:nNF,4) = sim_data.Tsh(sNF_begin:sNF_end);
    data(nNF+1:nF,4) = sim_data.Tsh(sF_begin:sF_end);
    
    data(n:nNF,5) = sim_data.Vexp_real(sNF_begin:sNF_end);
    data(nNF+1:nF,5) = sim_data.Vexp_real(sF_begin:sF_end);
    
    data(n:nNF,6) = sim_data.Tsup_real(sNF_begin:sNF_end);
    data(nNF+1:nF,6) = sim_data.Tsup_real(sF_begin:sF_end);
    
    data(n:nNF,7) = sim_data.Tsuc_real(sNF_begin:sNF_end);
    data(nNF+1:nF,7) = sim_data.Tsuc_real(sF_begin:sF_end);
    
    data(n:nNF,8) = sim_data.Tret_real(sNF_begin:sNF_end);
    data(nNF+1:nF,8) = sim_data.Tret_real(sF_begin:sF_end);
    
    data(n:nNF,9) = sim_data.Tdis_real(sNF_begin:sNF_end);
    data(nNF+1:nF,9) = sim_data.Tdis_real(sF_begin:sF_end);
    
    data(n:nNF,10) = sim_data.Psuc_real(sNF_begin:sNF_end);
    data(nNF+1:nF,10) = sim_data.Psuc_real(sF_begin:sF_end);
    
    data(n:nNF,11) = sim_data.Pdis_real(sNF_begin:sNF_end);
    data(nNF+1:nF,11) = sim_data.Pdis_real(sF_begin:sF_end);
    
    data(n:nNF,12) = sim_data.EvapFanSpeed_real(sNF_begin:sNF_end);
    data(nNF+1:nF,12) = sim_data.EvapFanSpeed_real(sF_begin:sF_end);
    
    data(n:nNF,13) = sim_data.Cpr_real(sNF_begin:sNF_end);
    data(nNF+1:nF,13) = sim_data.Cpr_real(sF_begin:sF_end);
    
    data(n:nNF,14) = sim_data.CondFanSpeed_real(sNF_begin:sNF_end);
    data(nNF+1:nF,14) = sim_data.CondFanSpeed_real(sF_begin:sF_end);
    
    data(n:nNF,15) = 0;
    data(nNF+1:nF,15) = 1;
    
    data(n:nF,16) = sim_data.HeatLoad;
    data(n:nF,17) = sim_data.Tset;
    
    data(n:nNF,18) = sim_data.CprScale(sNF_begin:sNF_end,2);
    data(nNF+1:nF,18) = sim_data.CprScale(sF_begin:sF_end,2);
    n = nF+1;
end 
% sorted_data = sortrows(data);
% end_data(1,1:16) = ["Test_nr","Faulty","T0","Tc","Tsh","Vexp_real","Tsup_real",...
%     "Tsuc_real","Tret_real","Tdis_real","EvapFanSpeed_real","Cpr_real",...
%     "CondFanSpeed_real"];
end_data(1,1:18) = ["Test_nr","T_0","T_c","T_sh","V_exp","T_sup",...
    "T_suc","T_ret","T_dis","P_suc","P_dis","EvapFanSpeed",...
    "Cpr","CondFanSpeed","Faulty","HeatLoad", "T_set", "Cpr_Scale"];
end_data(2:n,1:18) = data;

%% Write CSV file
save = [parDir '\data\allNoisyData2.csv'];
writematrix(end_data,save)