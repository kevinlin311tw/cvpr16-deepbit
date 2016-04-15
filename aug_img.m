function aug_img()
addpath(genpath(pwd));
fprintf('cvpr16-deepbit startup\n');

Set = 'cifar10';
txtfile1 = sprintf('./data/cifar10/%s_train_a.txt',Set);
fid1 = fopen(txtfile1, 'wt');
txtfile2 = sprintf('./data/cifar10/%s_train_b.txt',Set);
fid2 = fopen(txtfile2, 'wt');

augmentation('/data/cifar10/train-file-list.txt',fid1,fid2);

fclose(fid1);
fclose(fid2);


end

function augmentation(list_im,fid1,fid2)

if ischar(list_im)
    %Assume it is a file contaning the list of images
    filename = list_im;
    list_im = read_cell(filename);
end

current_path = pwd;
DataDir = '/rotations/';
PatchDir2 = sprintf('%s%s/', DataDir);

[data_num x] = size(list_im);

for i = 1:data_num
    fprintf('[augmentation] rotating image %d\n',i);
    img_path = list_im{i};
    img_dirs = strsplit(img_path,'/');
    %img = imread(img_path);
    %imshow(query_img);
    for aug = 1:5
        PatchDir = sprintf('%s/data/cifar10/rotations/%s_%d', pwd, img_dirs{4}, aug);
        if exist(PatchDir, 'dir') == 0
            mkdir(PatchDir);
        end
        f = sprintf('%s/%s',PatchDir,img_dirs{5});
        rotate_angle = 5*(aug-3);
        %img2 = imrotate(img,rotate_angle);
        %imshow(img2);
        %[x,y] = size(img2);
        %img3 = imcrop(img2,[(x-32)/2 (x-32)/2 31 31]);
        %imshow(img3);
        %imwrite(img3,f);
        fprintf(fid1, '%s%s 1\n',pwd,img_path);
        weight = exp(1)^(-(aug-3)*(aug-3)/2);
        fprintf(fid2, '%s %f\n',f, weight);
     end
end


end

