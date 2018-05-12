%% Feature extraction code
% Movie lens dataset contain 1,000,209 anonymous ratings of 3952 movies 
% made by 6,040 MovieLens users who joined MovieLens in 2000.
% It returns Train matrix, Test Matrix for subset of Movie features and Y for learning
% theta^* from linear regression.

function [W_train,W_train_d,W_test,movie_features,Y,all_movie_features,all_Y] = feature_extraction_subset(L,d,selected_movies,selected_users,Adventure_movie_id)

num_users = 6040;
num_movies = 3952;

fid = fopen('dataset/ratings.dat','r');
datacell = textscan(fid, '%d::%d::%d::%d'); % UserID::MovieID::Rating::Timestamp
fclose(fid);

num_ratings = length(datacell{1});

W_feedback_matrix = zeros(num_users , num_movies);
W_feedback_matrix_ratings = zeros(num_users , num_movies);


%% Creating Feedback Matrix. Setting it 1 if user rated movie with more than 3 stars.

for i = 1:num_ratings
    user_id = datacell{1}(i);
    movie_id = datacell{2}(i);
    rating = datacell{3}(i);
    
    if (any(Adventure_movie_id(:) == movie_id))
        W_feedback_matrix_ratings(user_id,movie_id) = rating;

        if rating > 3
           W_feedback_matrix(user_id,movie_id) = 1; 
        end
    end
   
end

% finding average rating for each movie from user rating
all_Y  = (mean(W_feedback_matrix_ratings,1)/5)';
all_Y = all_Y(Adventure_movie_id);
%% Randomly permuting users

W_feedback_matrix = W_feedback_matrix(selected_users,Adventure_movie_id);

%% Splitting into Train and Test matrix

W_train = W_feedback_matrix(1:num_users/2,:);
W_test = W_feedback_matrix(num_users/2+1:end,:);

% Saving train and test matrix
% save('W_train','W_train');
% save('W_test','W_test');


%% Performing Collaborative filtering to learn features of movies

% all_Y = mean(W_train,1)';

[U,S,V] = svd(W_train);

W_train_d = U(:,1:d)*S(1:d,1:d)*V(:,1:d)'; % rank-d approximation 
% fprintf('Rank d = %d reconstruction error is: %f \n',d,norm(W_train-W_train_d));

all_movie_features = V(:,1:d)*S(1:d,1:d);

%% Splitting into Train and Test matrix of size (num_users/2 X L)

W_train = W_feedback_matrix(1:num_users/2,1:L);
W_test = W_feedback_matrix(num_users/2+1:end,1:L);

movie_features = all_movie_features(1:L,:);
Y = all_Y(1:L);

end



