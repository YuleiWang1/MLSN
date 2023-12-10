clear
%load fpcasegundo
%load Segundo

%load fpcasandiego100
%load Sandiego100

%load fpcasandiego2
%load Sandiego2

load fpcaurban
load Urban

%load fpcabeach
%load Beach
%load groundtruth

%load fpcahydice
%load HYDICE

%pca = mean(firstpca,3);
r = 2; % try r=2, 4, or 8    Segundo=12/28最好, Sandiego100=10, Sandiego2=1, Urban=2, HYDICE=1/2最好, beach=3
eps = 0.2^2; % try eps=0.1^2, 0.2^2, 0.4^2   double(groundtruth)
%q = guidedfilter(pca, detect, r, eps);
pca = firstpca(:,:,1);  % Segundo=1, Sandiego100=2, Urban=1, Sandiego2=1， HYDICE=3, beach=1
%pca = double(groundtruth);
%a = max(max(pca(:,:,1)));
%b = min(min(pca(:,:,1)));
%pca = (pca-b)/(a-b);
%q = guidedfilter_color(firstpca, detect, r, eps);
q = guidedfilter(pca, detect, r, eps);
%{
[a, b] = find(q > 1);
for i = 1:size(b)
  q(a(i),b(i)) = 1;
end
%}

%{
H = size(detect,1);
W = size(detect,2);
q = reshape(q, [1,H*W]);
max1 =max(q);
min1 = min(q);
q = (q - min1)/(max1 - min1);
q = reshape(q,[H, W]);
%}
q =  hyperNormalize( q );
%save ('D:\Workspace\meta_learning_workspace\painting\datasets\Segundo\guideq','q');
%save ('D:\Workspace\meta_learning_workspace\Siamese_network_ based_on_meta_learning\guide_image_filter\guideq','q');


%save ('D:\Workspace\meta_learning_workspace\painting\datasets\Sandiego100\guideq','q');
%save ('D:\Workspace\meta_learning_workspace\Siamese_network_ based_on_meta_learning\guide_image_filter\guideq','q');

%save ('D:\Workspace\meta_learning_workspace\painting\datasets\Urban\guideq','q');
%save ('D:\Workspace\meta_learning_workspace\Siamese_network_ based_on_meta_learning\guide_image_filter\guideq','q');

%save ('D:\Workspace\meta_learning_workspace\painting\datasets\Sandiego2\guideq','q');
%save ('D:\Workspace\meta_learning_workspace\Siamese_network_ based_on_meta_learning\guide_image_filter\guideq','q');

%save ('D:\Workspace\meta_learning_workspace\painting\datasets\Beach\guideq','q');
%save ('D:\Workspace\meta_learning_workspace\Siamese_network_ based_on_meta_learning\guide_image_filter\guideq','q');

save ('D:\Workspace\meta_learning_workspace\painting\datasets\HYDICE_urban\guideq','q');
save ('D:\Workspace\meta_learning_workspace\Siamese_network_ based_on_meta_learning\guide_image_filter\guideq','q');


%q = 1000.^q;
%a = max(max(q));
%b = min(min(q));
%q = (q-b)/(a-b);

imagesc(q)

