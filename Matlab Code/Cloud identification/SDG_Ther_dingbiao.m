function [b]=SDG_Ther_dingbiao(I)
    [h,w]=size(I(:,:,1));
    I(I==min(I))=nan;
    b(:,:,1)=double(I(:,:,1))*0.003947+0.167126;
    b(:,:,2)=double(I(:,:,2))*0.003946+0.124622;
    b(:,:,3)=double(I(:,:,3))*0.005329+0.222530;
    b(b==min(b))=nan;
 end
