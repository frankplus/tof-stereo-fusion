%% TODO: da gestire le maschere (opportuno crop dei dati riproiettatti)

% Pack the dataset for the CNN training and testing
% Data augmentation is applied to the training set, see the code and the
% other comments for more details

clear all
close all

rng('default')

%Folder containing the ToF and Stereo data
reprojected_dataset_root = '/home/francesco/Desktop/Tesi/tof-stereo-fusion/reprojected_data/';

% Name for the output .mat file containing the output dataset (e.g. dataset.mat)
output_name = 'dataset';

% size of the patches extracted in each image for the training set (patch_size x patch_size)
patch_size = 128;
% number of the patches extracted in each image for the training set
num_patch_x_Im = 30;
% ratio if training set data used for validatoin
validation_ratio = 0.2;

% remove not valid points in the reprojected data. Used for vaidation and
% test data but not used in the training ones (patch_size x patch_size)
crop_size = 20;

%% Create training set structure
training_data = [];
training_label = [];

% selected validation scenes, they are not to be included in the training
% set
validation_set_indices = randperm(40, ceil(40*validation_ratio))-1;

disp('Generate training set and apply data augmentation')
num_im = 0;
index_history = [];
for i = 0:39
    if isempty(find(validation_set_indices-i==0)) == 1
        
        load(strcat(reprojected_dataset_root,'train/stereo_disparity_maps/scene_',num2str(i,'%05.4u'),'_disparity_stereo_dx.mat'),'disparity_stereo_dx')
        load(strcat(reprojected_dataset_root,'train/tof_disparity_reprojected/scene_',num2str(i,'%05.4u'),'_MHz120_disparity_reprojected.mat'),'disparity_120_PU_rp')
        load(strcat(reprojected_dataset_root,'train/amplitude_reprojected/scene_',num2str(i,'%05.4u'),'_MHz120_amplitude_reprojected.mat'),'amplitude_120_rp')
        load(strcat(reprojected_dataset_root,'train/disparity_ground_truth_halfResolution/scene_',num2str(i,'%05.4u'),'_disparity_stereo_dx_GT.mat'),'disparity_GT')
        
        feature = cat( 3, disparity_stereo_dx, disparity_120_PU_rp, amplitude_120_rp);
        
        % Crop data to remove not valid points
        feature = feature(crop_size+1:end-crop_size,crop_size+1:end-crop_size,:);
        
        % Ground truth depth map
        label = disparity_GT(crop_size+1:end-crop_size,crop_size+1:end-crop_size,:);
        
        % Apply data augmentation to the input and the labels used to train
        % the CNN
        [curr_training_data, curr_training_label, index_history] = data_augmentation(feature,label, patch_size,num_patch_x_Im, index_history, num_im);
     
        % add data augmeted CNN input and label to the training set
        training_data(:,:,:,sum(index_history(1:end-1))+1:sum(index_history))=single(curr_training_data);
        training_label(:,:,:,sum(index_history(1:end-1))+1:sum(index_history))=single(curr_training_label);
        num_im = num_im+1;
    end
end
disp('... Done')

%% Create test set structure
disp('Generate test set...')
index = 1;

test_data = [];
test_label = [];

for i = 0:14
    feature = [];
    
    load(strcat(reprojected_dataset_root,'test/stereo_disparity_maps/test_scene_',num2str(i,'%05.4u'),'_disparity_stereo_dx.mat'),'disparity_stereo_dx')
    load(strcat(reprojected_dataset_root,'test/tof_disparity_reprojected/test_scene_',num2str(i,'%05.4u'),'_MHz120_disparity_reprojected.mat'),'disparity_120_PU_rp')
    load(strcat(reprojected_dataset_root,'test/amplitude_reprojected/test_scene_',num2str(i,'%05.4u'),'_MHz120_amplitude_reprojected.mat'),'amplitude_120_rp')
    load(strcat(reprojected_dataset_root,'test/disparity_ground_truth_halfResolution/test_scene_',num2str(i,'%05.4u'),'_disparity_stereo_dx_GT.mat'),'disparity_GT')
    
    feature = cat( 3, disparity_stereo_dx, disparity_120_PU_rp, amplitude_120_rp);
    
    % Crop data to remove not valid points
    feature = feature(crop_size+1:end-crop_size,crop_size+1:end-crop_size,:);
    disparity_GT = disparity_GT(crop_size+1:end-crop_size,crop_size+1:end-crop_size);
    
    test_data(:,:,:,i+1)=single(feature);
    test_label(:,:,:,i+1)=single(disparity_GT);
end
disp('... Done')
%% Create validation set structure
disp('Generate validation set...')
validation_full_data = [];
validation_full_label = [];
num_im = 1;
for i = validation_set_indices
    feature = [];
    
    load(strcat(reprojected_dataset_root,'train/stereo_disparity_maps/scene_',num2str(i,'%05.4u'),'_disparity_stereo_dx.mat'),'disparity_stereo_dx')
    load(strcat(reprojected_dataset_root,'train/tof_disparity_reprojected/scene_',num2str(i,'%05.4u'),'_MHz120_disparity_reprojected.mat'),'disparity_120_PU_rp')
    load(strcat(reprojected_dataset_root,'train/amplitude_reprojected/scene_',num2str(i,'%05.4u'),'_MHz120_amplitude_reprojected.mat'),'amplitude_120_rp')
    load(strcat(reprojected_dataset_root,'train/disparity_ground_truth_halfResolution/scene_',num2str(i,'%05.4u'),'_disparity_stereo_dx_GT.mat'),'disparity_GT')
    
    feature = cat( 3 ,disparity_stereo_dx ,disparity_120_PU_rp, amplitude_120_rp);

    % Crop data to remove not valid points
    feature = feature(crop_size+1:end-crop_size,crop_size+1:end-crop_size,:,:);
    disparity_GT = disparity_GT(crop_size+1:end-crop_size,crop_size+1:end-crop_size);
    
    validation_full_data(:,:,:,num_im)=single(feature);
    validation_full_label(:,:,:,num_im)=single(disparity_GT);
    num_im = num_im+1;
end
disp('... Done')
% Save the training, validation and test data

% data reshape required in the reading process in python (tables package)
training_data = permute(training_data,[3 2 1 4]);
training_label= permute(training_label,[3 2 1 4]);

test_data= permute(test_data,[ 3 2 1 4]);
test_label= permute(test_label,[ 3 2 1 4]);

validation_full_data= permute(validation_full_data,[3 2 1 4]);
validation_full_label= permute(validation_full_label,[ 3 2 1 4]);

disp('Save data...')
save(strcat(output_name,'.mat'),'-v7.3',...
    'training_data','training_label',...
    'test_data','test_label','validation_full_data','validation_full_label')
disp('... Done')

function [training_data, training_label, index_history] = data_augmentation(feature,label,patch_size,num_patch_x_Im, index_history, num_im)
% data augmetation consists in 
% 1) random sampling of the patches
% 2) image flipping left/right and up/down
% 3) image rotation 5 dg and -5dg

% INPUT:
%   feature = float tensor, tensor containing the CNN input data 
%   label = float tensor, tensor containing the CNN lables (related to feature)
%   patch_size = integer, size of the patches extracted in each image for the training set (patch_size x patch_size)
%   num_patch_x_Im = integer, number of patches extracted in each image for the training set
%   index_history = vector of integers, how many patches for processed
%                   scenes have been added to the training set
%   num_im = integer, number of scenes currently added to the training set
% OUTPUT:
%   training_data = float tensor, data augmented version of "feature"
%   training_label = float tensor, data augmented version of "label"
%   index_history = vector of integers,updated version of the input "index_history"

aus_r = randperm(size(feature,1)-patch_size+1, num_patch_x_Im);
aus_s = randperm(size(feature,2)-patch_size+1, num_patch_x_Im);

aux_feature = [];
aux_label = [];

index = 1;
% patches from standard image
aux_feature1 = feature;
aux_label1 = label;

for u = 1:num_patch_x_Im
    r = aus_r(u);
    s = aus_s(u);
    aux_feature(:,:,:,index) =  aux_feature1(r:r+patch_size-1,s:s+patch_size-1,:,1);
    aux_label(:,:,:,index) = aux_label1(r:r+patch_size-1,s:s+patch_size-1,:,1);
    index = index+1;
end

% patches from up-down flipped image
aux_feature1 = flipud(feature);
aux_label1 = flipud(label);

for u = 1:num_patch_x_Im
    r = aus_r(u);
    s = aus_s(u);
    aux_feature(:,:,:,index) =  aux_feature1(r:r+patch_size-1,s:s+patch_size-1,:,1);
    aux_label(:,:,:,index) = aux_label1(r:r+patch_size-1,s:s+patch_size-1,:,1);
    index = index+1;
end

% patches from left-right flipped image
aux_feature1 = fliplr(feature);
aux_label1 = fliplr(label);

for u = 1:num_patch_x_Im
    r = aus_r(u);
    s = aus_s(u);
    aux_feature(:,:,:,index) =  aux_feature1(r:r+patch_size-1,s:s+patch_size-1,:,1);
    aux_label(:,:,:,index) = aux_label1(r:r+patch_size-1,s:s+patch_size-1,:,1);
    index = index+1;
end

% -5deg rotated images
aux_feature1 = imrotate(feature,-5);
aux_label1 = imrotate(label,-5);

for u = 1:num_patch_x_Im
    r = aus_r(u);
    s = aus_s(u);
    patch = aux_feature1(r:r+patch_size-1,s:s+patch_size-1,:,1);
    if min(min(patch(:,:,1)))>0
        aux_feature(:,:,:,index) =  patch;
        aux_label(:,:,:,index) = aux_label1(r:r+patch_size-1,s:s+patch_size-1,:,1);
        index = index+1;
    end
end

% +5deg rotated images
aux_feature1 = imrotate(feature,5);
aux_label1 = imrotate(label,5);

for u = 1:num_patch_x_Im
    r = aus_r(u);
    s = aus_s(u);
    patch = aux_feature1(r:r+patch_size-1,s:s+patch_size-1,:,1);
    if min(min(patch(:,:,1)))>0
        aux_feature(:,:,:,index) =  patch;
        aux_label(:,:,:,index) = aux_label1(r:r+patch_size-1,s:s+patch_size-1,:,1);
        index = index+1;
    end
end


index_history(num_im+1) = index-1;

training_data=single(aux_feature);
training_label=single(aux_label);
end
