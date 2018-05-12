% Runs RankedLinTS with base algorithm as CascadeLinTS and returns Cumulative Regret, Cumulative Reward and Top K movies
function [regret,reward,A] = RankedLinTS(variance,d,n,K,movie_features,W_test,A_star,theta_star,w_movie)

% Initialization
regret =zeros(n,1);
reward =zeros(n,1);

% Running different bandit for each movie from 1 to K
for k = 1:K
    struct_k(k).iM = eye(d);
    struct_k(k).B = zeros(d,1);
    struct_k(k).theta_bar = zeros(d ,1); 
    struct_k(k).theta = zeros(d ,1); 
end
   

% Looping through horizon n
for t = 1:n
    A = zeros(K,1);
    % for all k items 
    for k = 1:K
        if issymmetric(struct_k(k).iM) == 0
            struct_k(k).iM =  triu(struct_k(k).iM) + triu(struct_k(k).iM)' - diag(struct_k(k).iM).*eye(d);
        end
        struct_k(k).theta_bar = (struct_k(k).iM*struct_k(k).B)/variance;
        
       % Sampling from Multivariate normal distribution
        struct_k(k).theta = mvnrnd(struct_k(k).theta_bar,struct_k(k).iM,1)';
        
        
        % Recommending a list of K items         
        A_temp = zeros(K,1);
        all_features = movie_features*struct_k(k).theta;
        for i = 1:(k-1)
            all_features(A(i)) = -100;
        end
        [~,A_temp] = sort(all_features,'descend');
        A(k) = A_temp(1);
    end
    
    % Getting feedback
    c_t = 100000;
    selected_user = randi([1 length(W_test)],1,1);
    for j = 1:K
       if W_test(selected_user,A(j)) == 1
         c_t = j;
         reward(t) = 1;
         break;
       end
    end
    
    % Calculating Regret
    reg = 0;
    for j = 1:min(c_t,K)
       e = A(j);
       e_star = A_star(j);
       reg = reg + w_movie(e_star) - w_movie(e);
    end
    regret(t) = reg;
    
    % Updating Statistics
    for k = 1:min(c_t,K)
       e = A(k);
       struct_k(k).iM = struct_k(k).iM - ((struct_k(k).iM*movie_features(e,:)'*movie_features(e,:)*struct_k(k).iM)/((movie_features(e,:)*struct_k(k).iM*movie_features(e,:)') + variance));
       if (j == c_t)
        struct_k(k).B = struct_k(k).B + movie_features(e,:)';
       end
    end
   
end
end