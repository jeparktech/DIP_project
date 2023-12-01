%%
addpath("IQA/CSV-master/Code/")
addpath("IQA/SUMMER-master/Code/")
addpath("IQA/UNIQUE-Unsupervised-Image-Quality-Estimation-master/")
addpath("IQA/MS-UNIQUE-master/")
CSV_PATH = "./Datasets/mini-CURE-OR/train.csv";
%%
process_folder('CURE-OR', 'NLM10', data_dict);
process_folder('CURE-OR', 'NLM20', data_dict);
process_folder('CURE-OR', 'NLM30', data_dict);

%%
csvfile = fopen(CSV_PATH, 'r');
header = fgetl(csvfile);
data_dict = containers.Map;
while ~feof(csvfile)
         line = fgetl(csvfile);
         values = strsplit(line, ',');
         image_id = values{1};
         challenge_type = values{4};
         challenge_level = values{5};
 
         % Create a dictionary entry with image ID as the key and other columns as the value
         data_dict(image_id) = struct('challenge_type', challenge_type, 'challenge_level', challenge_level);
end
 
     % Close the CSV file
     fclose(csvfile);

%%
% Main function to process the folder
function process_folder(dataset, method, data_dict)
    output_path = './Outputs';
    
    if strcmp(method, 'bilateral')
        csv_filename = fullfile(output_path, 'Bilateral_results_matlab.csv');
        denoised_img = 'Bilateral.png';
    elseif strcmp(method, 'NLM10')
        csv_filename = fullfile(output_path, 'NLM10_results_matlab.csv');
        denoised_img = 'NLM10.png';
    elseif strcmp(method, 'NLM20')
        csv_filename = fullfile(output_path, 'NLM20_results_matlab.csv');
        denoised_img = 'NLM20.png';
    elseif strcmp(method, 'NLM30')
        csv_filename = fullfile(output_path, 'NLM30_results_matlab.csv');
        denoised_img = 'NLM30.png';
    else
        csv_filename = '';
        denoised_img = '';
    end

    % Open the CSV file for writing
    csvfile = fopen(csv_filename, 'a');
    % Write header to CSV file
    %fprintf(csvfile, 'Dataset,Challenge,Noise Type,UNIQUE,MS-UNIQUE,CSV,SUMMER\n');
    % Process each subfolder
    subfolders = dir(fullfile(output_path, '*/'));
    for subfolder = subfolders'
        subfolder_path = fullfile(subfolder.folder, subfolder.name);
        
        if contains(subfolder_path, dataset)
             if strcmp(dataset, 'CURE-OR')
                 parts = strsplit(subfolder_path, filesep);
                 folder_name = parts{end};
                 if ~contains(folder_name, '.') && ~contains(folder_name, 'DS')
                    id = num2str(round(str2double(folder_name)));
                    disp(id)
                    level = data_dict(id).challenge_level;
                    type = data_dict(id).challenge_type;
                    image_files = dir(fullfile(subfolder_path, '*.png'));
                    for image_file = image_files'
                      if strcmp(image_file.name, denoised_img)
                      denoised_image_path = fullfile(image_file.folder, denoised_img);
                      gt_image_path = fullfile(image_file.folder, 'GT.png');
                      disp(denoised_image_path)
                      denoised_image = imread(denoised_image_path);
                      gt_image = imread(gt_image_path);
                      unique = mslUNIQUE(gt_image,denoised_image);
                      ms_unique = mslMSUNIQUE(gt_image,denoised_image);
                      summer = SUMMER(gt_image,denoised_image);
                      csv_val = csv(gt_image,denoised_image);
 
%                       % Write results to CSV
                      fprintf(csvfile, '%s,%s,%s,%f,%f,%f,%f\n', dataset, level, type, unique, ms_unique, csv_val, summer);
                      end
                    end
                 end
                 
            elseif strcmp(dataset, 'CURE-TSR')
                  parts = strsplit(subfolder_path, filesep);
                  folder_name = parts{end};
                  if contains(folder_name, '_') && ~contains(folder_name, 'DS')
                      disp(folder_name)
                    comp = strsplit(folder_name, '_');
                  noise_type = comp{2};
                  challenge = comp{3};
                  image_files = dir(fullfile(subfolder_path, '*.png'));
                  for image_file = image_files'
                      if strcmp(image_file.name, denoised_img)
                      denoised_image_path = fullfile(image_file.folder, denoised_img);
                      gt_image_path = fullfile(image_file.folder, 'GT.png');
                      disp(denoised_image_path)
                      denoised_image = imread(denoised_image_path);
                      gt_image = imread(gt_image_path);
                      unique = mslUNIQUE(gt_image,denoised_image);
                      ms_unique = mslMSUNIQUE(gt_image,denoised_image);
                      summer = SUMMER(gt_image,denoised_image);
                      csv_val = csv(gt_image,denoised_image);
 
%                       % Write results to CSV
                      fprintf(csvfile, '%s,%s,%s,%f,%f,%f,%f\n', dataset, challenge, noise_type, unique, ms_unique, csv_val, summer);
                      end
                  end
                  end
                  
             
             elseif strcmp(dataset, 'SIDD')
                 image_folders = dir(fullfile(subfolder_path, '*/'));
                 for image_folder = image_folders'
                     folder_name = image_folder.name;
                     if strcmp(folder_name, denoised_img)
                         denoised_image_path = fullfile(image_folder.folder, folder_name);
                         gt_image_path = fullfile(image_folder.folder, 'GT.png');
 
                         denoised_image = imread(denoised_image_path);
                         gt_image = imread(gt_image_path);
 
                         unique = mslUNIQUE(gt_image,denoised_image);
                         ms_unique = mslMSUNIQUE(gt_image,denoised_image);
                         summer = SUMMER(gt_image,denoised_image);
                         csv_val = csv(gt_image,denoised_image);
 
                         fprintf(csvfile, '%s,-1,-1,%f,%f,%f,%f\n', dataset, unique, ms_unique, csv_val, summer);
 
                     end
                 end
             end
        end
    end

    % Close the CSV file
    fclose(csvfile);
end
%%
% 
% function data_dict = read_csv_and_create_dict(csv_file_path)
%     data_dict = containers.Map;
% 
%     % Open the CSV file for reading
%     csvfile = fopen(csv_file_path, 'r');
% 
%     % Read header
%     header = fgetl(csvfile);
% 
%     % Read data
%     while ~feof(csvfile)
%         line = fgetl(csvfile);
%         values = strsplit(line, ',');
%         image_id = values{1};
%         challenge_type = values{2};
%         challenge_level = values{3};
% 
%         % Create a dictionary entry with image ID as the key and other columns as the value
%         data_dict(image_id) = struct('challenge_type', challenge_type, 'challenge_level', challenge_level);
%     end
% 
%     % Close the CSV file
%     fclose(csvfile);
% end
%%