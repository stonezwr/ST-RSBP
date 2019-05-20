function output = eij_step(input_num,output_num,input_time,output_time, T_REF, TAU_M, TAU_S, end_time)
%EIJ_STEP Summary of this function goes here
%   Detailed explanation goes here
    if input_num == 0 || output_num == 0
        output = 0;
        return;
    end
	output = 0;
	p=0;
	q=0;
	index_in=1;
	index_out=1;
	t_ref=0;
	for t = 1:1:end_time
        p=p-p/TAU_S;
        if index_in<=input_num && input_time(index_in)==t-1
			p=p+1;
			index_in=index_in+1;
        end
		q = q-q/TAU_M+p/TAU_S;
		if t_ref~=0
			q=0;
			t_ref=t_ref-1;
        end
		if output_time(index_out)==t
			output = output+q;
			index_out = index_out+1;
			t_ref=T_REF;
			q=0;
        end
		if index_out>output_num
			break;
        end
    end
end

        