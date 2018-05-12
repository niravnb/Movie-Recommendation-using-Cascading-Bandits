%% Figure 2a: The n-step regret of CascadeLinTS for varying number of features d.
clc; clear all; 
close all;

%% Initializing variables
num_movies = 256;
variance = 1;
n = 100000;
K = 4;
d_values = [10,20,40];

% Loading randomly selected user & movies to reproduce figures
load('selected_movies.mat');
load('selected_users.mat');

% In case want to run on any random movies and users
% selected_movies = randi([1 3952],1,num_movies);
% selected_users = randperm(linspace(1,6040,1));

c = 1;
color = ['r','g','b'];

%% Looping for different values of d

for d = d_values

if d == 10
    load('selected_movies_10.mat');
    load('selected_users_10.mat');
else
    load('selected_movies.mat');
    load('selected_users.mat');
end
    
%% Feature Extraction 
[W_train,W_train_d,W_test,movie_features,Y,all_movie_features,all_Y] = feature_extraction(num_movies,d,selected_movies,selected_users);
movie_features = normr(movie_features); % Normalizing movie features to norm 1
   
    % Finding theta_star and A_star
    mdf = fitlm(movie_features,Y,'linear'); % Performing Linear regression
    theta_star = table2array(mdf.Coefficients(2:end,1));
    w_movie = movie_features*theta_star;
    [s,in] = sort(w_movie,'descend');
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
    SEM = std(cum_regret')/sqrt(itr);   % Standard Error
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
print('-djpeg','Movie_Fig_2a.jpg', '-r300');
print('-dpng','Movie_Fig_2a.png', '-r300');
savefig('Movie_Fig_2a.fig')
close all;
