function KP = computeKP_D(nodeXY1,nodeXY2,sigma, wd, row_index2, dMax)
% KP is the node similarity matrix.
% Here is the radial kernel


    
r = size(nodeXY1,2);
c = size(nodeXY2,2);

KP = zeros(r,c);

if dMax == 0.
   D = zeros(r,c);
   for i = 1:r
    for j = 1:c
        
        node1 = nodeXY1(:, i);
        node2 = nodeXY2(:, j);
        D(i,j) =norm(node1 - node2);

    end
   end
   dMax = max(max(D));
   
end

for i = 1:r
    for j = 1:c
        if ismember(i, row_index2)

            KP(i,j) = 1 - wd/dMax;
        else
            node1 = nodeXY1(:, i);
            node2 = nodeXY2(:, j);
            KP(i,j) = 1-norm(node1 - node2)/dMax;
        end
    end
end


for i = 1:r
    for j = 1:c
        if ismember(i, row_index2)

            KP(i,j) = 1 - wd/dMax;
        else
            node1 = nodeXY1(:, i);
            node2 = nodeXY2(:, j);
            KP(i,j) = 1-norm(node1 - node2)/dMax;
        end
    end
end
%     minKP = min(KP(:));
%     if minKP<0
%         %fprintf('the minimum of KQ is negative: %d\n',minKQ);
%         KP = KP + 2*abs(minKP);
%     end
KP = sigma * KP;


stat = [quantile(KP(:),0.01) quantile(KP(:),0.25) quantile(KP(:),0.5) quantile(KP(:),0.75) quantile(KP(:),0.99)];
fprintf('node affinity 5 number summary: %.2f %.2f %.2f %.2f %.2f\n', stat');
