%% Performing offline recommendation using linear regression on Train matrix
% Result: Total cumulative reward is: 74565 
% Accuracy of offline recommendation from linear regression is: 74.565000 

clc; clear all; 
close all;

%% Initializing variables
num_movies = 256;
variance = 1;
n = 100000;
K = 4;
d = 20;

% Loading randomly selected user & movies to reproduce figures
load('selected_movies.mat');
load('selected_users.mat');

% In case want to run on any random movies and users
% selected_movies = randi([1 3952],1,num_movies);
% selected_users = randperm(linspace(1,6040,1));

c = 1;
color = ['r','g','b'];

%% Feature Extraction 
[W_train,W_train_d,W_test,movie_features,Y,all_movie_features,all_Y] = feature_extraction(num_movies,d,selected_movies,selected_users);
movie_features = normr(movie_features); % Normalizing movie features to norm 1
   
% Finding theta_star and A_star
mdf = fitlm(movie_features,Y,'linear'); % Performing Linear regression
theta_star = table2array(mdf.Coefficients(2:end,1));
w_movie = movie_features*theta_star;
[s,in] = sort(w_movie,'descend');
A_star = in(1:K);


%% Showing A_Star found from linear regression and finding it's accuray on test matrix

reward = zeros(n,1);
for i = 1:n
selected_user = randi([1 length(W_test)],1,1);
   for j = 1:K
       if W_test(selected_user,A_star(j)) == 1
         reward(i) = 1;
         break;
       end
   end
end
    
fprintf('Total cumulative reward is: %d \n', sum(reward));
fprintf('Accuracy of offline recommendation from linear regression is: %f \n',(sum(reward)/n)*100);