clear 
close all

%% new comments

%% ER graph
N=600;%no of Node
p=0.03;%connection Probability
G = gsp_erdos_renyi(N,p);
W=full(G.W);
D=diag(sum(W));
L=D-W;
[Q, LAMBDA]=eig(L);
lambda=sum(LAMBDA);

%% parameter Set up
p_lamda=0.01;
p_R=0.01;

k=5;
alpha_star=[ones(1,k) zeros(1,N-k)]';

p_C=sqrt(alpha_star'*LAMBDA*alpha_star);
[~,max_node]=max(Q*alpha_star);
Reward_max=Q(max_node,:)*alpha_star;

p_delta=0.001;
p_LAMDA=p_lamda*diag(ones(1,N));

p_gamma=1;
K=N/2;
T=250;

p_beta=p_R*sqrt(log(2*K*(1+log2(T))/p_delta))+p_C;
A_temp=Q(1:K,:)';

%total step 
% 
% %effective dimension
% d=1:N;
% d=((d-1).*lambda)<T/(log(1+T/p_lamda));
% [~, d]=find(d, 1, 'last' );

alpha_temp=zeros(N,1);
J=ceil(log2(T));
tj=2.^(0:J);


for j=1:J 
    V_temp=p_gamma*LAMBDA+p_LAMDA;
    
    for t=tj(j):min(tj(j+1), T)
        %node select
        inv_v=inv(V_temp);
        c_mul=sqrt(sum((A_temp'*inv_v).*A_temp',2));
        [~, s_node(t)]= max(c_mul);
        %reward
        r(t)=Q(s_node(t),:)*alpha_star;
        V_temp=V_temp+A_temp(:, s_node(t))*A_temp(:, s_node(t))';
        
        Regret(t)=Reward_max*t-sum(r);
        
    end 
    inv_v=inv(V_temp);
    alpha_temp=inv_v*A_temp(:,s_node(tj(j):t))*r(tj(j):t)';
    c_mul=sqrt(sum((A_temp'*inv_v).*A_temp',2));
    p=max(((alpha_temp'*A_temp)'-p_beta*sqrt(sum((A_temp'*inv_v).*A_temp',2))));
    
    s_index=find(((alpha_temp'*A_temp)'+p_beta*sqrt(sum((A_temp'*inv_v).*A_temp',2)))>=p);
    
    A_temp=A_temp(:,s_index);    
end





