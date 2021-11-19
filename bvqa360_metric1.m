modelfun1 = @(b,x)(b(2)+((b(1)-b(2))./(1+exp(-(x-b(3))/abs(b(4))))));
modelfun2 = @(beta,x)(beta(1)*(0.5-1./(1+exp(beta(2)*(x-beta(3))))) + beta(4)*x+beta(5)); 
modelfun3 = @(b,x)(b(1) + b(2)*x.^(2));

%%% ==============================================================================================================================================================
%%% Our dataset of ODV-VQA %%%
%%% ==============================================================================================================================================================
% Method param-b PLCC SPCC KRCC RMSE MAE
% % OURS 0.006 0.9209 0.9236 0.7760 4.6165 3.1136
b=0.006;
X = textread('J:\bvqa360_compared_algo\OURS\ours_dmos.txt');
y = textread('J:\bvqa360_compared_algo\NSTSS\GT_dmos.txt');
Y = y(:)';
allbeta=nlinfit(X',Y,modelfun1,[1 0 mean(X) b],statset('MaxIter',10000000000000));
X_fit=modelfun1(allbeta,X);


alldiff=[corr(Y',X_fit,'type','Pearson')...
        corr(Y',X_fit,'type','Spearman')...
        corr(Y',X_fit,'type','Kendall')...
        100*sqrt(sum((Y-X_fit').^2)/length(Y))...%RMSE
        100*mean(abs(Y-X_fit'));
        ];
%fprintf(fid1, '%8.3f\t', alldiff);
%fclose(fid1);
disp(alldiff) 

[xx, id] = sort(X);
yy = X_fit(id); 
yys = Y(id);

plot(xx, yys, 'xb','MarkerSize',7.5);
hold on;
plot(xx, yy, 'r', 'LineWidth',1.4 );
xlabel('Score', 'FontWeight','bold');
ylabel('DMOS', 'FontWeight','bold');
set(gca,'fontname','times','FontWeight','bold')  % Set it to times
set(gca,'FontSize',20, 'FontWeight','bold');



