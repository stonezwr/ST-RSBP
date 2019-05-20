num_oi=1:1:50;
num_oj=1:1:50;
T_REF=2;
end_time=800;
TAU_M = 64;
TAU_S = 8;
iteration = 1000;
mean_final = [];
for input_num=num_oj
    mean_tmp = [];
    for output_num=num_oi
        E_ij = [];
        for i=1:1:iteration
            input_time = random_time(input_num, end_time, T_REF);
            output_time = random_time(output_num, end_time, T_REF);
            eij = eij_step(input_num,output_num,input_time,output_time, T_REF, TAU_M, TAU_S, end_time);
            E_ij = [E_ij, eij];
        end
        mean_tmp = [mean_tmp, mean(E_ij)];
    end 
    mean_final = [mean_final; mean_tmp];
end
figure;
mesh(num_oi,num_oj,mean_final); xlabel('output num'); ylabel('input num'); title('means of new effect ratio \partial e_{ij}/\partial o_i');
p_final = [];
for in=num_oj
    a=mean_final(in,:);
    p = polyfit(num_oj,a,4);
    p_final = [p_final; p];    
end
save('p_Tau_64_800.txt','p_final','-ascii');
