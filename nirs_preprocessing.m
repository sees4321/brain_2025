%% Spectratech OEG-16 device의 HbO, HbR 추정
% data load
clear all; close all; clc;


% wavelength = [840 770];
DataType = {'emo','mist','d2','nback'};


for CurrentData = 1:4

if CurrentData == 1 || CurrentData ==2
    DataDir = fullfile('D:\연구\4. 뇌선도\NIRS\day1)emotion+mist');
    cd(DataDir)
else
    DataDir = fullfile('D:\연구\4. 뇌선도\NIRS\day2)nback+d2');
    cd(DataDir)
end
a = ls;
a(1:2,:)=[];

%%

fs = 1/0.081/2; 

[b_pre,a_pre] = butter(3, 0.5/fs*2,'low');
[b_post,a_post] = butter(3, [0.01 0.2]/fs*2,'bandpass'); 
fNIRS_data = [];
for ii = 1:size(a,1)
    C = [];
    C_deoxy = [];
    C_oxy = [];
    C_total = [];
    data = [];
    trig = [];

    cd([DataDir '\' a(ii,:)])
    Data_list = ls;
    Data_list(1:2,:)=[];
    if CurrentData ==2 && ii == 6
        continue;
    end
    if contains(Data_list(1,:),[DataType{CurrentData} '.mat'])||contains(Data_list(2,:),[DataType{CurrentData} '.mat'])
        load([DataDir '\' a(ii,:) '\' DataType{CurrentData} '.mat'])
    else
        continue
    end
    ChNummber = (1:36); % 전체 채널: Source(6)*Detector(6)
    nCh = length(ChNummber);
    
    ch_3 = [1 7 2 8 9 14 15 21 16 22 23 28 29 35 30 36]; % S-D 3cm 인 채널만 추출
    
    % OEG SpO2 Applied Technology Edition V1.3 문서 10p 기반 HbO, HbR 계산
    e = [692.36 1022 ;
        1311.88 650 ];

    trig = data(:,3);
    
    raw_high = data(:,4:nCh+3); % total oxy ch
    raw_low  = data(:,nCh+4:end); % total deoxy ch
    
    raw_3high = raw_high(:,ch_3);
    raw_3low = raw_low(:,ch_3);

    raw = cat(2,raw_3high,raw_3low);

    raw = filtfilt(b_pre,a_pre,raw);

    raw_3high = raw(:,1:size(ch_3,2));
    raw_3low = raw(:,size(ch_3,2)+1:end);
    
    ref_high = mean(raw_3high,1); % example data array: ((current time(1)+sample time(1)+trigger(1)+770nm(36ch)+840nm(36ch)) x time
    ref_low  = mean(raw_3low,1);
    
    o_high = real(-log10(raw_3high./ref_high));
    o_low  = real(-log10(raw_3low./ref_low));
    
    for i = 1:length(ch_3)
        ch = i
        for t  = 1:size(o_high,1)
            C = inv(e)*[o_high(t,i);o_low(t,i)];
            C_total{t,i} = C;
            % cm-1/M -> mMol*mm
            C_deoxy(t,i) = C(1)*10000;
            C_oxy(t,i)   = C(2)*10000;
        end
    end

    fNIRS_final = cat(2,C_oxy,C_deoxy);

    fNIRS_final = filtfilt(b_post,a_post,fNIRS_final);

    fNIRS_data.data{ii} = cat(2,trig,fNIRS_final);
    fNIRS_data.ID(ii) = cellstr(a(ii,:));

end
save(['D:\연구\4. 뇌선도\NIRS\' DataType{CurrentData} '.mat'],"fNIRS_data")
end