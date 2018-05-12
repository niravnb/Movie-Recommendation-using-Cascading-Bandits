%% Figure 2b: The n-step regret of CascadeLinTS in a subset movies for varying number of features d.
clc; clear all; 
close all;

%% Initializing variables
num_movies = 256;
variance = 1;
n = 100000;
K = 4;
d_values = [10,20,40];

% Loading dataset 
fid = fopen('dataset/movies.dat','r');
movies_datacell = textscan(fid, '%d::%s','whitespace', '','Delimiter','\n'); % MovieID::Title::Genres
fclose(fid);

% Finding subset of movies based on Genres
Subset_movie_id = [];
for i = 1:length(movies_datacell{1})
    
    if(contains(movies_datacell{2}(i),'Romance')) % or Drama
        Subset_movie_id = [Subset_movie_id movies_datacell{1}(i)];
    end

end



% Loading randomly selected user & movies to reproduce figures
load('selected_movies_subset.mat');
load('selected_users_subset.mat');

% In case want to run on any random movies and users
% selected_movies = Adventure_movie_id(1,randperm(length(Adventure_movie_id))); %randi([1 3952],1,num_movies);
% selected_movies = selected_movies(1:num_movies);
% selected_users = randperm(linspace(1,6040,1));

c = 1;
color = ['r','g','b'];

%% Looping for different values of d

for d = d_values

if d == 10
    load('selected_movies_subset_10.mat');
    load('selected_users_subset_10.mat');
else
    load('selected_movies_subset.mat');
    load('selected_users_subset.mat');
end
    
%% Feature Extraction 
[W_train,W_train_d,W_test,movie_features,Y,all_movie_features,all_Y] = feature_extraction_subset(num_movies,d,selected_movies_subset,selected_users_subset,Subset_movie_id);
movie_features = normr(movie_features); % Normalizing movie features to norm 1


   
    % Finding theta_star and A_star
    mdf = fitlm(movie_features,Y,'linear'); % Performing Linear regression
    theta_star = table2array(mdf.Coefficients(2:end,1));
    w_movie = movie_features*theta_star;
    [s,in] = sort(w_movie,'descend');
    w_movie(w_movie<0) = 0;
    A_star = in(1:K);
    
    % Looping for 10 iterations and collecting cumulative regret
    itr = 10;
    cum_regret = zeros(n,itr);

    for i = 1:itr
        [regret,reward,A] = CascadeLinTS(variance,d,n,K,movie_features,W_test,A_star,theta_star,w_movie);
  
        cum_regret(:,i) = cumsum(regret);
    end


    % Plotting average cummulative regret
     avg_cum_regret = mean(cum_regret,2);

    x = linspace(1,n,n);
    SEM = std(cum_regret')/sqrt(itr);    % Standard Error
    [r,h(c)]= boundedline(x,avg_cum_regret,SEM,['-',color(c)]);
    set(get(get(h(c),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    c = c+1;
    hold on
    
    
end

title(['L = ',num2str(num_movies),', K = ',num2str(K)],'FontWeight','bold','FontSize',15);
xlabel('Step n','FontWeight','bold','FontSize',15);
ylabel('Regret','FontWeight','bold','FontSize',15);
lgd = legend('d = 10','d = 20','d = 40');
lgd.FontWeight = 'bold';
lgd.FontSize = 15;
lgd.Location = 'northwest';
box on
print('-djpeg','Subset_Movie_Fig_2b.jpg', '-r300');
print('-dpng','Subset_Movie_Fig_2b.png', '-r300');
savefig('Subset_Movie_Fig_2b.fig')
close all;
