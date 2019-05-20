%function my_Lyon_ear

% My own function for passing speeches by Lyon passive ear model
% Preprocessing all data ti46 (train and test)

Type_digits = '00010203040506070809';
Type_20 = '00010203040506070809energohpnorbrpspstys';
Type_alpha = '0a0b0c0d0e0f0g0h0i0j0k0l0m0n0o0p0q0r0s0t0u0v0w0x0y0z';

Sex = 'fm'; % 0: female 1: male
ind = [1, 2, 3, 4 ,5 ,6, 7, 8];


h = waitbar(0,'Initializing waitbar...');
%Training data
type=2;
if(type == 1)
%         Type = 'ti_digits';
    num_samples = 10;
    for i=1:1:num_samples
        filepath=sprintf('Ti46_digits/test/%d',i-1);
        mkdir(filepath);
        filepath=sprintf('Ti46_digits/train/%d',i-1);
        mkdir(filepath);
    end
else
%         Type = 'ti_alpha';
   num_samples = 26;
   for i=1:1:num_samples
        filepath=sprintf('data_new/ti_alpha/test/%d',i-1);
        mkdir(filepath);
        filepath=sprintf('data_new/ti_alpha/train/%d',i-1);
        mkdir(filepath);
    end
end
for iii = 1:1:3
    total_num=num_samples*2*8*10;
    convert_num=0;
    for s = 1:2
        sex = Sex(s);
        for j = 1:8
            person = ind(j);
            for i = 1:num_samples
                for k = 1:10
                    convert_num=convert_num+1;
                    if(type == 1)
                        if(iii==1)
                            filename1 = sprintf('TI46/ti20/train/%s%d/%s%s%s%dset%d.wav',sex,person,Type_20(2*i-1),Type_20(2*i),sex,person,k-1);
                        elseif(iii==2)
                            filename1 = sprintf('TI46/ti20/test/%s%d/%s%s%s%ds%dt0.wav',sex,person,Type_20(2*i-1),Type_20(2*i),sex,person,k);
                        else
                            filename1 = sprintf('TI46/ti20/test/%s%d/%s%s%s%ds%dt1.wav',sex,person,Type_20(2*i-1),Type_20(2*i),sex,person,k);
                        end
                        a = exist(filename1);
                        if(a == 0)
                            continue;
                        end
                        tap = audioread(filename1);
                        y = LyonPassiveEar(tap,12500,200);
                        if(iii==1)
                            filename2 = sprintf('Ti46_digits/train/%d/%s%d_u%d_c%d.dat',i-1,sex,person,k,i-1);
                        elseif(iii==2)
                            filename2 = sprintf('Ti46_digits/test/%d/t0_%s%d_u%d_c%d.dat',i-1,sex,person,k,i-1);
                        else
                            filename2 = sprintf('Ti46_digits/test/%d/t1_%s%d_u%d_c%d.dat',i-1,sex,person,k,i-1);
                        end
                        save (filename2,'y','-ascii');
                        waitbar(convert_num/total_num, h, sprintf('Processing ti20:%d %.2f%% ...',iii,convert_num*100/total_num));
                    else
                        if(iii==1)
                            filename1 = sprintf('TI46/ti_alpha/train/%s%d/%s%s%s%dset%d.wav',sex,person,Type_alpha(2*i-1),Type_alpha(2*i),sex,person,k-1);
                        elseif(iii==2)
                            filename1 = sprintf('TI46/ti_alpha/test/%s%d/%s%s%s%ds%dt0.wav',sex,person,Type_alpha(2*i-1),Type_alpha(2*i),sex,person,k);
                        else
                            filename1 = sprintf('TI46/ti_alpha/test/%s%d/%s%s%s%ds%dt1.wav',sex,person,Type_alpha(2*i-1),Type_alpha(2*i),sex,person,k);
                        end
                        a = exist(filename1);
                        if(a == 0)
                            continue;
                        end
                        tap = audioread(filename1);
                        y = LyonPassiveEar(tap,12500,200);
                        if(iii==1)
                            filename2 = sprintf('data_new/ti_alpha/train/%d/%s%d_u%d_c%d.dat',i-1,sex,person,k,i-1);
                        elseif(iii==2)
                            filename2 = sprintf('data_new/ti_alpha/test/%d/t0_%s%d_u%d_c%d.dat',i-1,sex,person,k,i-1);
                        else
                            filename2 = sprintf('data_new/ti_alpha/test/%d/t1_%s%d_u%d_c%d.dat',i-1,sex,person,k,i-1);
                        end
                        save (filename2,'y','-ascii');
                        waitbar(convert_num/total_num, h, sprintf('Processing ti_alpha:%d %.2f%% ...',iii, convert_num*100/total_num));
                    end
                end
            end
        end
    end
end
close(h);
