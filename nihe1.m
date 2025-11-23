%绘图
x = [0,1,2.5];
y = [0,1,2];
plot(x,y,'o')
%最小二乘法算k和b
n = size(x,1);
k = (n*sum(x.*y)-sum(x)*sum(y))/(n*sum(x.*x)-sum(x)*sum(x));
b = (sum(x.*x)*sum(y)-sum(x)*sum(x.*y))/(n*sum(x.*x)-sum(x)*sum(x));

% 输出k和b值
fprintf('最小二乘法结果：\n');
fprintf('斜率 k = %.4f\n', k);
fprintf('截距 b = %.4f\n', b);