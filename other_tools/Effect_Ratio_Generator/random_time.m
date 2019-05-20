function output = random_time(num, end_time, T_REF)
%RANDOM_TIME Summary of this function goes here
%   Detailed explanation goes here
record=0;
output = [];
while(record<num)
    t=round(rand(1,1)*end_time);
    exist=0;
    for i=0:1:T_REF
        if ismember(t-i,output)||ismember(t+i,output)
            exist=1;
            break;
        end
    end
    if exist==1
        continue;
    end
    output = [output, t];
    record = record+1;
end
output=sort(output);
end

