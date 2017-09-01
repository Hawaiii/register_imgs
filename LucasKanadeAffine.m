function M = LucasKanadeAffine(It, It1)
% Output:
%  M: affine transformation matrix such that (xt1 yt1 1)' = M*(xt yt 1)', 
%     [1+a b c; d 1+e f; 0 0 1];
p = zeros(6,1);
dp = ones(6,1);

% Compute gradient
[grad_ItX, grad_ItY] = gradient(It);
grad_ItX = reshape(grad_ItX, 1, []);
grad_ItY = reshape(grad_ItY, 1, []);
% Compute Jacobian
J = zeros(2, size(p,1), size(It,1)*size(It,2));
for x = 1:size(It,2)
    J(1,1,(x-1)*size(It,1)+1:x*size(It,1)) = x;
    J(2,4,(x-1)*size(It,1)+1:x*size(It,1)) = x;
end
for x = 1:size(It,2)
    J(1,2,(x-1)*size(It,1)+1:x*size(It,1)) = 1:size(It,1);
    J(2,5,(x-1)*size(It,1)+1:x*size(It,1)) = 1:size(It,1);
end
J(1,3,:) = 1;
J(2,6,:) = 1;
% Compute Hessian
H = zeros(size(p,1), size(p,1));
dTJ = zeros(size(J,3), size(p,1));
for i = 1:size(J,3)
    dTJ(i,:) = [grad_ItX(i) grad_ItY(i)]*J(:,:,i);
    H = H + dTJ(i,:)'*dTJ(i,:);
end
invH = inv(H);

% The warp
M = eye(3);
M(1,:) = M(1,:)+p(1:3)';
M(2,:) = M(2,:)+p(4:6)';

while sumsqr(dp) > 10^(-3)
    % 1. Warp I with W(x;p): I(W(x;p))
    tform = affine2d(M');
    RA = imref2d([size(It1,1) size(It1,2)], [1 size(It1,2)], [1 size(It1,1)]);
    It1_warp = imwarp(It1, tform, 'OutputView', RA);
    WarpANDIt = im2bw(It1_warp.*It,0);
    
    % 2. Compute error image T(x)-I(W(x;p))
    diff = (It - It1_warp).*WarpANDIt;
    
    % 3. Compute dp
    dp = invH * dTJ'*reshape(diff,[],1);
    
    % 4.Update W(x;p)=W(x;p)+inv(W(x;dp))
    dM = eye(3);
    dM(1,:) = dM(1,:)+dp(1:3)';
    dM(2,:) = dM(2,:)+dp(4:6)';
    M = M * inv(dM);
end
M = inv(M);
end