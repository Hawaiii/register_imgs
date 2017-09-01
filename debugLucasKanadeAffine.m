% make image
I = im2double(imread('cameraman.tif'));
% rotate image
Ir = imrotate(I, 30,'crop');
% compute affine warp
M = LucasKanadeAffine(I, Ir);
% warp image
tform = affine2d(M');
RA = imref2d([size(I,1) size(I,2)], [1 size(I,2)], [1 size(I,1)]);
Iwarp = imwarp(I, tform, 'OutputView', RA);
% compare
figure;imshow(Ir);title('Ir');
figure;imshow(Iwarp);title('Iwarp');