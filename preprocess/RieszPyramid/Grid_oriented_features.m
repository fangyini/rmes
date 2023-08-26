function [OF_Hist] = Grid_oriented_features(Input_Mag,Input_Angle,Feat_Dim,numBins)
    
    [h,w] = size(Input_Mag);
    Ang_Limits =  linspace(0+pi/numBins,2*pi-pi/numBins,numBins);
    Ang_Lim1 = Ang_Limits(1:numBins-1);
    Ang_Lim2 = Ang_Limits(2:end);
    
    OF_Hist = zeros(1,round(numBins*prod(Feat_Dim)));

    ROI_Box = [1,w,1,h];
    X_Limit = (linspace(ROI_Box(1),ROI_Box(2),Feat_Dim(1)+1));
    Y_Limit = (linspace(ROI_Box(3),ROI_Box(4),Feat_Dim(2)+1));
    X_Lim1 = X_Limit(1:Feat_Dim(1));
    X_Lim2 = X_Limit(2:end);
    Y_Lim1 = Y_Limit(1:Feat_Dim(2));
    Y_Lim2 = Y_Limit(2:end);
    [XX,YY]=meshgrid(1:w,1:h);
    row = YY(:);
    col = XX(:);
    for i=1:length(row)
        Xquad = find((col(i)-X_Lim1>=0) & (X_Lim2-col(i)>0));
        Yquad = find((row(i)-Y_Lim1>=0) & (Y_Lim2-row(i)>0));
        Ang_can = Input_Angle(row(i),col(i));
        if sign(Ang_can)<0
            Ang_can = 2*pi+Ang_can;
        end
        Ang_quad = find(((Ang_can-Ang_Lim1)>=0) & ((Ang_Lim2-Ang_can)>0))+1;
        if isempty(Ang_quad)
            if Ang_can<Ang_Limits(1) || Ang_can>Ang_Limits(end)
                Ang_quad = 1;
            end
        end
        if length(Xquad)>1||length(Yquad)>1
            disp('hello')
        end
        index = Ang_quad + numBins*((Xquad-1)+(Yquad-1)*Feat_Dim(1));
        OF_Hist(index) = OF_Hist(index)+Input_Mag(row(i),col(i)); 
    end
    tot = sum(OF_Hist(:));
    OF_Hist = OF_Hist/tot;
end