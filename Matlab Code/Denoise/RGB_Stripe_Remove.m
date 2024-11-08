 function [moban]=RGB_Stripe_Remove(I,theta)

a=abs(max(theta)-min(theta))/0.001;
[R,~]=radon(I,theta);
[h,w]=size(I);
si=size(R);
N=si(2);
m=si(1);
dd=I;
ij=1;
moban=ones(h,w);

while ij<a
      [peaks,loc]=findpeaks(R(:,ij),'Threshold',60);
      szj=(size(R)-1)/2;

      k=tan(theta(ij)*pi/180);

      k2=tan((90-theta(ij))*pi/180);
      for iw=1:size(peaks)
     if peaks(iw)>2000
       loc(iw)=find(R(:,ij)==peaks(iw))-szj(1);
       xx=loc(iw)*cos(theta(ij)*pi/180);
       yy=loc(iw)*sin(theta(ij)*pi/180);

       xxx=xx+w/2;
       yyy=h/2-yy;
       b=yyy-k2*xxx;

       for j=1:h
              x(iw,j)=round((j-b)/k2);    
           if x(iw,j)>1 && x(iw,j)<w
              moban(j,(x(iw,j)))=0;
              moban(j,(x(iw,j))-1)=0;
           end
       end

     end
       end
    ij=ij+1;
end

 end