clc
clear

imput=""; % Input file
[I,R]=geotiffread(imput);

DD=SDG_Ther_dingbiao(I);
Tk=SDG_Ther_ChangetoT(DD);

Tk1=Tk(:,:,1);
Tk2=Tk(:,:,2);
Tk3=Tk(:,:,3);

tk2_1=Tk2-Tk1;
tk3_1=Tk3-Tk1;
tk3_2=Tk3-Tk2;
tk31=Tk3+Tk1;
tk32=Tk3+Tk2;
tk21=Tk2+Tk1;
i31=tk3_1./tk31;
i32=tk3_2./tk32;
i21=tk2_1./tk21;

II=[];
JJ=[];

t1=[];
t2=[];
t3=[];
t4=[];
t5=[];
t6=[];

[h,w]=size(I(:,:,1));

for i=1:h
    for j=1:w
        if isnan(Tk1(i,j))==0
            t1(end+1)=Tk1(i,j);
            t2(end+1)=Tk2(i,j);
            t3(end+1)=Tk3(i,j);
            t4(end+1)=i32(i,j);
            t5(end+1)=i31(i,j);
            t6(end+1)=i21(i,j);
            II(end+1)=i;
            JJ(end+1)=j;
        end
    end
end
t1=t1';
t2=t2';
t3=t3';
t4=t4';
t5=t5';
t6=t6';

data=[t1,t2,t3,t4,t5,t6];

k = 10; 

[idx, C] = kmeans(data, k);

LeiBie=zeros(h,w);
for i=1:length(idx)
    LeiBie (II(i),JJ(i))=idx(i);
end

info = geotiffinfo(imput);
output3='';
geotiffwrite(output3,LeiBie,R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag)
