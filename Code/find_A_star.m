% For finding optimal Top K list.
% Input Feedback Matrix with 0,1 entries
% Row represents user, Columns Represent movie items
% Output K list containing top K movies that must be attracted by most of the users.

function A_star = find_A_star(W , K)
    A_star = [];
    W1 = linspace(1,size(W,2),size(W,2));
    W = [W1 ; W];
    while(K > 0 && size(W ,1) ~= 1)
        %K , size(W ,1), W
        Y = W(2:end,: );
        [value, index] = max(sum(Y, 1));       % get index of max column
        A_star = [A_star , W(1 ,index)];
        X=[];       %temp matrix
        X = [X ; W(1,:)];
        for i = 1:size(W ,1)
            if W(i,index) == 0
                X = [X ; W(i,:)];
            end
        end
        if size(X ,1) ==1
            break;
        end
        X(:,index)=[];      %removing max index column  
        W = X;      % replacing the matrix
        
        K = K-1;
    end
    

