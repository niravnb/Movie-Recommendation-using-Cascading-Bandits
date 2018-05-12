%UCB like algorithm for cascading bandits, it returns cumulative regret,
%cumulative reward and top K movies

function [regret,reward,A] = CascadeUCB(n,K,num_movies,W_test,A_star,w_movie)
% intialization

% W_movie as true mean for each movie
L = num_movies;
w_cap = binornd(1,w_movie);
T = zeros(L,1);
U = zeros(L,1);
regret =zeros(n,1);
reward =zeros(n,1);

% Looping through horizon n
for i = 1:n
    for e = 1:L
        U(e) = w_cap(e) +  sqrt((1.5*log(i))/T(e)); % Calculating UCB of each movie
    end
    % Recommending a list of top K items 
    [sorted_U ,A] = sort(U,'descend');
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
       temp_reg = w_movie(e_star) - w_movie(e);
       reg = reg + temp_reg;       
    end
    regret(i) = reg;
    
   % Updating Statistics
   for j = 1:min(c_t,K)
       e = A(j);
       T(e) = T(e) + 1;
       if (j == c_t)
          w_cap(e) = ((T(e) - 1)*w_cap(e) + 1)/T(e);
       end
   end
   
end
end