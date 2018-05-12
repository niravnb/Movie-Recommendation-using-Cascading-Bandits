%% Figure 1b: Comparison of n-step regret of baselines at L= 16 .
clc; clear all; 
close all;

%% Initializing variables
num_movies = 3952;
d = 20;
variance = 0.01;
n = 100000;
K = 4;

%% Loading randomly selected user & movies
num_movies = 3020;
selected_movies = randi([1 3952],1,num_movies);
selected_users = randperm(linspace(1,6040,1));


%% Feature Extraction 
[W_train,W_train_d,W_test,movie_features,Y,all_movie_features,all_Y] = feature_extraction(num_movies,d,selected_movies,selected_users);
movie_features = normr(movie_features);
all_movie_features = normr(all_movie_features);


%% Looping for different algorithms
c = 1;
color = ['r','b','g','m'];

for algo_no = [1,2,3,4]
   
    % Finding theta_star and A_star
    mdf = fitlm(movie_features,Y,'linear');
    theta_star = table2array(mdf.Coefficients(2:end,1));
    w_movie = movie_features*theta_star;
    [s,in] = sort(w_movie,'descend');
    A_star = in(1:K);
    
    % Looping for 10 iterations
    itr = 10;
    cum_regret = zeros(n,itr);

    for i = 1:itr
        if algo_no == 1
        % selected_movies=load('selected_movie_user_31.mat');
        [regret,reward,A] = CascadeLinTS(variance,d,n,K,movie_features,W_test,A_star,theta_star,w_movie);
        end
        if algo_no == 2
        % selected_movies=load('selected_movie_user_32.mat');
        [regret,reward,A] = CascadeLinUCB(variance,d,n,K,movie_features,W_test,A_star,theta_star,w_movie);
        end
        if algo_no == 3
        % selected_movies=load('selected_movie_user_33.mat');
        [regret,reward,A] = RankedLinTS(variance,d,n,K,movie_features,W_test,A_star,theta_star,w_movie);
        end
        if algo_no == 4
        % selected_movies=load('selected_movie_user_34.mat');
        [regret,reward,A] = CascadeUCB(n,K,num_movies,W_test,A_star,w_movie);  
        end
        cum_regret(:,i) = cumsum(regret);
    end


    % Plotting average cummulative regret
     avg_cum_regret = mean(cum_regret,2);
     % cum_regret = selected_movies.selected_movies;
     x = linspace(1,n,n);
     SEM= std(cum_regret')/sqrt(itr);    
     [r,h(c)]= boundedline(x,cum_regret,10,['-',color(c)]);
     set(get(get(h(c),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
     c = c+1;
     hold on
    
    
end
title(['L = 3952, K = 4'],'FontWeight','bold','FontSize',15);
xlabel('Step n','FontWeight','bold','FontSize',15);
ylabel('Regret','FontWeight','bold','FontSize',15);
lgd = legend('CascadeUCB1','CascadeLinTS','RankedLinTS','CascadeLinUCB');
lgd.FontWeight = 'bold';
lgd.FontSize = 15;
lgd.Location = 'northwest';
box on
ylim([0 1400]);
print('-dpng','Fig_1c.png', '-r300');
close all;