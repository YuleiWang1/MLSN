%% Binarization
clear
%load B
%load groundtruth
%load Segundo
%load Sandiego100
%load Sandiego2
load Urban
%load Beach
%load HYDICE
load guideq
%m = find(groundtruth == 1);
%n = double(q(m));
%k = min(n);
%k = 0.9850;  % segundo=0.9910,Sandiego100=0.9850

se = strel('square',2);

%qo = imopen(q,se);     
qc = imclose(q,se);
%save ('D:\Workspace\meta_learning_workspace\painting\datasets\Sandiego100\result','qc');
%save ('D:\Workspace\meta_learning_workspace\painting\datasets\Sandiego2\result','qc');
save ('D:\Workspace\meta_learning_workspace\painting\datasets\Urban\result','qc');
%save ('D:\Workspace\meta_learning_workspace\painting\datasets\Segundo\result','qc');
%save ('D:\Workspace\meta_learning_workspace\painting\datasets\Beach\result','qc');
%save ('D:\Workspace\meta_learning_workspace\painting\datasets\HYDICE_urban\result','qc');

figure(1)
imagesc(qc)
%{
result_binary = qc;

[a, b] = find(result_binary > k);
for i = 1:size(b)
  result_binary(a(i),b(i)) = 1;
end
[c, d] = find(result_binary <1);
for i=1:size(d)
  result_binary(c(i),d(i)) = 0;
end

%save ('D:\Workspace\meta_learning_workspace\painting\datasets\Sandiego100\result_binary','result_binary');
save ('D:\Workspace\meta_learning_workspace\painting\datasets\Urban\result_binary','result_binary');
figure(2)
imagesc(result_binary)
%}

%figure(3)
%imagesc(groundtruth)