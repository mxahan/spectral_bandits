clear 
close all

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

V_temp=LAMBDA+p_LAMDA;

%total step 
T=250;
%effective dimension
d=1:N;
d=((d-1).*lambda)<T/(log(1+T/p_lamda));
[~, d]=find(d, 1, 'last' );

alpha_temp=zeros(N,1);

%% Implementation

for t=1:T
    c=2*p_R*sqrt(d*log(1+t/p_lamda)+2*log(1/p_delta))+p_C;
    inv_v=inv(V_temp);
    c_mul=sqrt(sum((Q*inv_v).*Q,2));
    [~, s_node(t)]=max(Q*alpha_temp+c*c_mul);
    %reward
    r(t)=Q(s_node(t),:)*alpha_star;
    %update
    V_temp=V_temp+Q(s_node(t),:)'*Q(s_node(t),:);
    alpha_temp=inv_v*(Q(s_node,:)'*r');
    
    Regret(t)=Reward_max*t-sum(r);
end




