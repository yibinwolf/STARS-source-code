function [b]=SDG_Ther_ChangetoT(I)
    b=[];
    b1=9.35;
    b2=10.73;
    b3=11.72;
    hh=6.626e-34;
    c=2.9979e8;
    k=1.3806e-23;
    d=2e24;
    [h,w]=size(I(:,:,1));
    for i=1:h
        for j=1:w
                b(i,j,1)=((1000000*hh*c)/(k*b1))/log(((d*hh*(c^2))/(I(i,j,1)*(b1^5)))+1);
                b(i,j,2)=((1000000*hh*c)/(k*b2))/log(((d*hh*(c^2))/(I(i,j,2)*(b2^5)))+1);
                b(i,j,3)=((1000000*hh*c)/(k*b3))/log(((d*hh*(c^2))/(I(i,j,3)*(b3^5)))+1);
        end
    end
    b(b==min(b))=nan;
 end
