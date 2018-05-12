%% Feature extraction code
% Movie lens dataset contain 1,000,209 anonymous ratings of 3952 movies 
% made by 6,040 MovieLens users who joined MovieLens in 2000.
% It returns Train matrix, Test Matrix, Movie features and Y for learning
% theta^* from linear regression.

function [W_train,W_train_d,W_test,movie_features,Y,all_movie_features,all_Y] = feature_extraction(L,d,selected_movies,selected_users)

% Initializing variables
num_users = 6040;
num_movies = 3952;

W_feedback_matrix = zeros(num_users , num_movies);
W_feedback_matrix_ratings = zeros(num_users , num_movies);

% Loading dataset
fid = fopen('dataset/ratings.dat','r');
datacell = textscan(fid, '%d::%d::%d::%d'); % UserID::MovieID::Rating::Timestamp
fclose(fid);

num_ratings = length(datacell{1});

%% Creating Feedback Matrix. Setting it 1 if user rated movie with more than 3 stars.

for i = 1:num_ratings
    user_id = datacell{1}(i);
    movie_id = datacell{2}(i);
    rating = datacell{3}(i);
    
    W_feedback_matrix_ratings(user_id,movie_id) = rating;
    
    if rating > 3
       W_feedback_matrix(user_id,movie_id) = 1; 
    end
   
end
% If want to find average rating for each movie from user rating from 1 to
% 5
% all_Y  = (mean(W_feedback_matrix_ratings,1)/5)';
%% Randomly permuting users

W_feedback_matrix = W_feedback_matrix(selected_users,:);

%% Splitting into Train and Test matrix

W_train = W_feedback_matrix(1:num_users/2,:);
W_test = W_feedback_matrix(num_users/2+1:end,:);

% for saving Train and test matrix
% save('W_train','W_train');
% save('W_test','W_test');


%% Performing Collaborative filtering to learn features of movies

all_Y = mean(W_train,1)';

[U,S,V] = svd(W_train); % SVD

W_train_d = U(:,1:d)*S(1:d,1:d)*V(:,1:d)'; % rank-d approximation 
fprintf('Rank d = %d reconstruction error is: %f \n',d,norm(W_train-W_train_d));

all_movie_features = V(:,1:d)*S(1:d,1:d); % Movies features of d dimension

%% Splitting into Train and Test matrix of size (num_users/2 X L)

W_train = W_feedback_matrix(1:num_users/2,selected_movies);
W_test = W_feedback_matrix(num_users/2+1:end,selected_movies);

movie_features = all_movie_features(selected_movies,:);
Y = all_Y(selected_movies);

end



