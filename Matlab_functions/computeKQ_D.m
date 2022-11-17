function KQ = computeKQ_D(beta1,beta2, elastic, sigma)

% KQ is the edge similarity matrix. Large value means more similarity
% Here is the SRVF inner product between two shapes


if elastic

%     r = size(beta1,3);c = size(beta2,3);
% 
%     KQ = zeros(2*r,2*c); % times 2 means directed graph
%     d_edge = [];
%     for i = 1:r
%         for j = 1:c
%             q1 = curve_to_q(beta1(:,:,i));
%             q2 = curve_to_q(beta2(:,:,j));
% %             [tmp,~] = ElasticShootingVector(q1,q2);
%             q2n = Find_Rotation_and_Seed_unique(q1,q2);
%             tmp = norm(q1 - q2n);
%             KQ(i,j) = tmp;
%         end
%     end
% 
%     % bottom right
%     for i = r+1:2*r
%         for j = c+1:2*c
%             KQ(i,j) = KQ(i-r,j-c);
%         end
%     end
% 
%     % top right, flip q2
%     c1 = c+1;c2 = 2*c;
%     for i = 1:r
%         for j = c1:c2
%             q1 = curve_to_q(beta1(:,:,i));
%             q2 = curve_to_q(fliplr(beta2(:,:,j-c)));
%             [tmp,~] = ElasticShootingVector(q1,q2);
%             q2n = Find_Rotation_and_Seed_unique(q1,q2);
%             tmp = norm(q1 - q2n);
%             d_edge = [d_edge, tmp];
%         end
%     end
% 
%     % bottom left, flip q1
%     for i = r+1:2*r
%         for j = 1:c
%             KQ(i,j) = KQ(i-r,j+c);
%         end
%     end
% 
%     minKQ = min(KQ(:));
%     if minKQ<0
%         %fprintf('the minimum of KQ is negative: %d\n',minKQ);
%         KQ = KQ + 2*abs(minKQ);
%     end
%     KQ = sigma * KQ;

    r = size(beta1,3);c = size(beta2,3);
    l1 = 1; l2 = 1;
%     for i=1:r
%         q = curve_to_q(beta1(:,:,i),false);
%         l1 = l1 + sqrt(InnerProd_Q(q,q));   
%     end
%     
%     for i =1:c
%         q = curve_to_q(beta2(:,:,i),false);
%         l2 = l2 + sqrt(InnerProd_Q(q,q));
%     end
        
    KQ = zeros(2*r,2*c); % times 2 means directed graph

    d_edge = [];
    for i = 1:r
        for j = 1:c
            q1 = curve_to_q(beta1(:,:,i));
            %             q1 = q1 / sqrt(InnerProd_Q(q1,q1));
            q2 = curve_to_q(beta2(:,:,j));
            %             q2 = q2 / sqrt(InnerProd_Q(q2,q2));
            q2 = Find_Rotation_and_Seed_unique(q1,q2);
            tmp = norm(q1/l1 - q2/l2);
            d_edge = [d_edge, tmp];
        end
    end
    c1 = c+1;c2 = 2*c;
    for i = 1:r
        for j = c1:c2
            q1 = curve_to_q(beta1(:,:,i));
%             q1 = q1 / sqrt(InnerProd_Q(q1,q1));
            q2 = curve_to_q(fliplr(beta2(:,:,j-c)));
%             q2 = q2 / sqrt(InnerProd_Q(q2,q2));
            q2 = Find_Rotation_and_Seed_unique(q1,q2);
            tmp = norm(q1/l1 - q2/l2);
            d_edge = [d_edge, tmp];
        end
    end
    
    dMax = max(d_edge);
    
    for i = 1:r
        for j = 1:c
            q1 = curve_to_q(beta1(:,:,i));
%             q1 = q1 / sqrt(InnerProd_Q(q1,q1));
            q2 = curve_to_q(beta2(:,:,j));
%             q2 = q2 / sqrt(InnerProd_Q(q2,q2));
            q2 = Find_Rotation_and_Seed_unique(q1,q2);
            tmp = 1 - norm(q1/l1 - q2/l2)/dMax;
            KQ(i,j) = tmp;
        end
    end

    % bottom right
    for i = r+1:2*r
        for j = c+1:2*c
            KQ(i,j) = KQ(i-r,j-c);
        end
    end

    % top right, flip q2
    c1 = c+1;c2 = 2*c;
    for i = 1:r
        for j = c1:c2
            q1 = curve_to_q(beta1(:,:,i));
%             q1 = q1 / sqrt(InnerProd_Q(q1,q1));
            q2 = curve_to_q(fliplr(beta2(:,:,j-c)));
%             q2 = q2 / sqrt(InnerProd_Q(q2,q2));
            q2 = Find_Rotation_and_Seed_unique(q1,q2);
            tmp = 1-norm(q1/l1 - q2/l2)/dMax;
            KQ(i,j) = tmp;
        end
    end

    % bottom left, flip q1
    for i = r+1:2*r
        for j = 1:c
            KQ(i,j) = KQ(i-r,j+c);
        end
    end

%     minKQ = min(KQ(:));
%     if minKQ<0
%         %fprintf('the minimum of KQ is negative: %d\n',minKQ);
%         KQ = KQ + 2*abs(minKQ);
%     end
    KQ = sigma * KQ;
    
else
    
    ita = 0;
    r = size(beta1,3);c = size(beta2,3);
    l1 = 1; l2 = 1;
%     for i=1:r
%         q = curve_to_q(beta1(:,:,i),false);
%         l1 = l1 + sqrt(InnerProd_Q(q,q));   
%     end
%     
%     for i =1:c
%         q = curve_to_q(beta2(:,:,i),false);
%         l2 = l2 + sqrt(InnerProd_Q(q,q));
%     end
        
    KQ = zeros(2*r,2*c); % times 2 means directed graph

    d_edge = [];
    for i = 1:r
        for j = 1:c
            q1 = curve_to_q(beta1(:,:,i));
            %             q1 = q1 / sqrt(InnerProd_Q(q1,q1));
            q2 = curve_to_q(beta2(:,:,j));
            %             q2 = q2 / sqrt(InnerProd_Q(q2,q2));
            tmp = L2Dist(q1/l1 , q2/l2)+ita*...
                abs(sqrt(InnerProd_Q(q1/l1, q1/l1)) - ...
                sqrt(InnerProd_Q(q2/l2, q2/l2)));
%             tmp=norm(q1/l1-q2/l2);
            d_edge = [d_edge, tmp];
        end
    end
    c1 = c+1;c2 = 2*c;
    for i = 1:r
        for j = c1:c2
            q1 = curve_to_q(beta1(:,:,i));
            %             q1 = q1 / sqrt(InnerProd_Q(q1,q1));
            q2 = curve_to_q(fliplr(beta2(:,:,j-c)));
            %             q2 = q2 / sqrt(InnerProd_Q(q2,q2));
            tmp = L2Dist(q1/l1 , q2/l2)+ita*...
                abs(sqrt(InnerProd_Q(q1/l1, q1/l1)) - ...
                sqrt(InnerProd_Q(q2/l2, q2/l2)));
            d_edge = [d_edge, tmp];
        end
    end
    
    dMax = max(d_edge);
    
    for i = 1:r
        for j = 1:c
            q1 = curve_to_q(beta1(:,:,i));
%             q1 = q1 / sqrt(InnerProd_Q(q1,q1));
            q2 = curve_to_q(beta2(:,:,j));
%             q2 = q2 / sqrt(InnerProd_Q(q2,q2));
            dd = L2Dist(q1/l1 , q2/l2)+ita*...
                abs(sqrt(InnerProd_Q(q1/l1, q1/l1)) - ...
                sqrt(InnerProd_Q(q2/l2, q2/l2)));
            tmp = 1 - dd/dMax;
%             tmp = 1 - norm(q1/l1-q2/l2)/dMax;
            KQ(i,j) = tmp;
        end
    end

    % bottom right
    for i = r+1:2*r
        for j = c+1:2*c
            KQ(i,j) = KQ(i-r,j-c);
        end
    end

    % top right, flip q2
    c1 = c+1;c2 = 2*c;
    for i = 1:r
        for j = c1:c2
            q1 = curve_to_q(beta1(:,:,i));
%             q1 = q1 / sqrt(InnerProd_Q(q1,q1));
            q2 = curve_to_q(fliplr(beta2(:,:,j-c)));
%             q2 = q2 / sqrt(InnerProd_Q(q2,q2));
            dd = L2Dist(q1/l1 , q2/l2)+ita*...
                abs(sqrt(InnerProd_Q(q1/l1, q1/l1)) - ...
                sqrt(InnerProd_Q(q2/l2, q2/l2)));
            tmp = 1 - dd/dMax;
%             tmp = 1 - norm(q1/l1-q2/l2)/dMax;
            KQ(i,j) = tmp;
        end
    end

    % bottom left, flip q1
    for i = r+1:2*r
        for j = 1:c
            KQ(i,j) = KQ(i-r,j+c);
        end
    end

%     minKQ = min(KQ(:));
%     if minKQ<0
%         %fprintf('the minimum of KQ is negative: %d\n',minKQ);
%         KQ = KQ + 2*abs(minKQ);
%     end
    KQ = sigma * KQ;
    
end

stat = [quantile(KQ(:),0.01) quantile(KQ(:),0.25) quantile(KQ(:),0.5) quantile(KQ(:),0.75) quantile(KQ(:),0.99)];
fprintf('edge affinity 5 number summary: %.2f %.2f %.2f %.2f %.2f\n', stat');
