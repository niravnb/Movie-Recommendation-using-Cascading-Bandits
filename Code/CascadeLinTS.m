% Runs CascadeLinTS and return Cumulative Regret, Cumulative Reward and Top K movies
function [regret,reward,A] = CascadeLinTS(variance,d,n,K,movie_features,W_test,A_star,theta_star,w_movie)

% Initialization
iM = eye(d);
B = zeros(d,1);
regret =zeros(n,1);
reward =zeros(n,1);

for i = 1:n
   % Sampling from Multivariate normal distribution 
   if issymmetric(iM) == 0
      iM =  triu(iM) + triu(iM)' - diag(iM).*eye(d);
   end
   
   theta_bar = (iM*B)/variance;
   theta = mvnrnd(theta_bar,iM,1)';
    
   % Recommending a list of K items 
   A = zeros(K,1);
   all_features = movie_features*theta;
   [~,A] = sort(all_features,'descend');
   A = A(1:K);


   % Getting feedback
   c_t = 10000;
   selected_user = randi([1 length(W_test)],1,1);
   for j = 1:K
       if W_test(selected_user,A(j)) == 1
         c_t = j;
         reward(i) = 1;
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
    regret(i) = reg;

%     Other way to find same regret as above    
%     rew = 1 - prod(1 - w_movie(A(1:min(c_t,K))));
%     rew_star =  1 - prod(1 - w_movie(A_star(1:min(c_t,K))));
%     regret(i) = rew_star - rew;

   
   % Updating Statistics
   for j = 1:min(c_t,K)
       e = A(j);
       iM = iM - ((iM*movie_features(e,:)'*movie_features(e,:)*iM)/((movie_features(e,:)*iM*movie_features(e,:)') + variance));
       if (j == c_t)
        B = B + movie_features(e,:)';
       end
   end
    
end


end